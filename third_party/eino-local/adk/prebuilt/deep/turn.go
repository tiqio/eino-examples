package deep

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"strings"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/schema"
)

type TurnResult struct {
	Turn        int
	MaxTurns    int
	Query       string
	LastContent string
	Accumulated string
	Interrupted bool
	HasError    bool
}

type TurnDecision struct {
	Continue   bool
	NextPrompt string
}

type TurnController struct {
	MaxTurns  int
	Decide    func(ctx context.Context, result TurnResult) TurnDecision
	BuildNext func(ctx context.Context, result TurnResult) string
	OnTurn    func(ctx context.Context, result TurnResult)
}

type turnResumableAgent[M adk.MessageType] struct {
	base adk.TypedResumableAgent[M]
	tc   *TurnController
}

func newTurnResumableAgent[M adk.MessageType](base adk.TypedResumableAgent[M], tc *TurnController) adk.TypedResumableAgent[M] {
	return &turnResumableAgent[M]{base: base, tc: tc}
}

func (a *turnResumableAgent[M]) Name(ctx context.Context) string { return a.base.Name(ctx) }
func (a *turnResumableAgent[M]) Description(ctx context.Context) string {
	return a.base.Description(ctx)
}

func (a *turnResumableAgent[M]) Run(ctx context.Context, input *adk.TypedAgentInput[M], options ...adk.AgentRunOption) *adk.AsyncIterator[*adk.TypedAgentEvent[M]] {
	out, gen := adk.NewAsyncIteratorPair[*adk.TypedAgentEvent[M]]()
	go func() {
		defer gen.Close()
		maxTurns := a.tc.MaxTurns
		if maxTurns <= 0 {
			maxTurns = 1
		}
		work := &adk.TypedAgentInput[M]{EnableStreaming: input.EnableStreaming}
		if input != nil {
			work.Messages = append(work.Messages, input.Messages...)
		}
		query := extractLastUserText(work.Messages)
		acc := ""
		for turn := 1; turn <= maxTurns; turn++ {
			log.Printf("[deep-turn] turn=%d/%d", turn, maxTurns)
			if shouldEmitTurnUI() {
				gen.Send(&adk.TypedAgentEvent[M]{
					Output: &adk.TypedAgentOutput[M]{
						MessageOutput: &adk.TypedMessageVariant[M]{
							IsStreaming: false,
							Message:     turnMarkerMessage[M](fmt.Sprintf("[Turn %d/%d] Continuing execution...", turn, maxTurns)),
							Role:        schema.Assistant,
						},
					},
				})
			}
			iter := a.base.Run(ctx, work, options...)
			last, interrupted, hasErr, ok := forwardAndCollectTyped(iter, gen)
			if !ok {
				return
			}
			if last != "" {
				if acc != "" {
					acc += "\n"
				}
				acc += last
			}
			result := TurnResult{Turn: turn, MaxTurns: maxTurns, Query: query, LastContent: last, Accumulated: acc, Interrupted: interrupted, HasError: hasErr}
			if a.tc.OnTurn != nil {
				a.tc.OnTurn(ctx, result)
			}
			if interrupted || hasErr {
				return
			}
			decision := defaultTurnDecision(ctx, a.tc, result)
			if !decision.Continue {
				return
			}
			if decision.NextPrompt != "" {
				work.Messages = append(work.Messages, any(userMessageForType[M](decision.NextPrompt)).(M))
			}
		}
	}()
	return out
}

func (a *turnResumableAgent[M]) Resume(ctx context.Context, info *adk.ResumeInfo, opts ...adk.AgentRunOption) *adk.AsyncIterator[*adk.TypedAgentEvent[M]] {
	return a.base.Resume(ctx, info, opts...)
}

func defaultTurnDecision(ctx context.Context, tc *TurnController, result TurnResult) TurnDecision {
	if tc == nil {
		return TurnDecision{Continue: false}
	}
	if tc.Decide != nil {
		return tc.Decide(ctx, result)
	}
	if tc.BuildNext != nil {
		next := tc.BuildNext(ctx, result)
		if next != "" {
			return TurnDecision{Continue: true, NextPrompt: next}
		}
	}
	return TurnDecision{Continue: false}
}

func forwardAndCollectTyped[M adk.MessageType](iter *adk.AsyncIterator[*adk.TypedAgentEvent[M]], gen *adk.AsyncGenerator[*adk.TypedAgentEvent[M]]) (string, bool, bool, bool) {
	last := ""
	interrupted := false
	hasErr := false
	for {
		e, ok := iter.Next()
		if !ok {
			return last, interrupted, hasErr, true
		}
		if e == nil {
			continue
		}
		if e.Output != nil && e.Output.MessageOutput != nil {
			captureTextFromTypedEvent(e, &last)
		}
		if e.Action != nil && e.Action.Interrupted != nil {
			interrupted = true
		}
		if e.Err != nil {
			hasErr = true
		}
		gen.Send(e)
	}
}

func captureTextFromTypedEvent[M adk.MessageType](e *adk.TypedAgentEvent[M], last *string) {
	mo := e.Output.MessageOutput
	if mo == nil {
		return
	}
	if !mo.IsStreaming {
		captureTypedMessageText(mo.Message, last)
		return
	}
	if mo.MessageStream == nil {
		return
	}
	cp := mo.MessageStream.Copy(2)
	if len(cp) != 2 {
		return
	}
	mo.MessageStream = cp[0]
	for {
		m, err := cp[1].Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			break
		}
		captureTypedMessageText(m, last)
	}
}

func captureTypedMessageText[M adk.MessageType](msg M, last *string) {
	if msg == nil {
		return
	}
	switch m := any(msg).(type) {
	case *schema.Message:
		if m != nil && m.Role == schema.Assistant && m.Content != "" {
			*last = m.Content
		}
	case *schema.AgenticMessage:
		if m == nil || m.Role != schema.AgenticRoleTypeAssistant {
			return
		}
		text := ""
		for _, b := range m.ContentBlocks {
			if b != nil && b.AssistantGenText != nil {
				text += b.AssistantGenText.Text
			}
		}
		if text != "" {
			*last = text
		}
	}
}

func userMessageForType[M adk.MessageType](text string) any {
	var zero M
	switch any(zero).(type) {
	case *schema.Message:
		return schema.UserMessage(text)
	case *schema.AgenticMessage:
		return schema.UserAgenticMessage(text)
	default:
		return nil
	}
}

func extractLastUserText[M adk.MessageType](messages []M) string {
	for i := len(messages) - 1; i >= 0; i-- {
		m := any(messages[i])
		switch v := m.(type) {
		case *schema.Message:
			if v != nil && v.Role == schema.User && v.Content != "" {
				return v.Content
			}
		case *schema.AgenticMessage:
			if v == nil || v.Role != schema.AgenticRoleTypeUser {
				continue
			}
			for _, b := range v.ContentBlocks {
				if b != nil && b.UserInputText != nil && b.UserInputText.Text != "" {
					return b.UserInputText.Text
				}
			}
		}
	}
	return ""
}

func turnMarkerMessage[M adk.MessageType](text string) M {
	var zero M
	switch any(zero).(type) {
	case *schema.Message:
		return any(schema.AssistantMessage(text, nil)).(M)
	case *schema.AgenticMessage:
		return any(&schema.AgenticMessage{Role: schema.AgenticRoleTypeAssistant, ContentBlocks: []*schema.ContentBlock{schema.NewContentBlock(&schema.AssistantGenText{Text: text})}}).(M)
	default:
		return zero
	}
}

func shouldEmitTurnUI() bool {
	v := strings.TrimSpace(os.Getenv("TURN_UI_ENABLED"))
	if v == "" {
		return true
	}
	return strings.EqualFold(v, "1") || strings.EqualFold(v, "true") || strings.EqualFold(v, "yes")
}
