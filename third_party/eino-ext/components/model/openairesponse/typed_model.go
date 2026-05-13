package openairesponse

import (
	"context"
	"errors"
	"io"
	"strings"

	"github.com/cloudwego/eino/components/model"
	"github.com/cloudwego/eino/schema"
)

type messageType interface {
	*schema.Message | *schema.AgenticMessage
}

type typedResponsesAPIModel[M messageType] struct {
	base *ResponsesAPIChatModel
}

func NewResponsesAPIModel[M messageType](ctx context.Context, config *ResponsesAPIConfig) (model.BaseModel[M], error) {
	base, err := NewResponsesAPIChatModel(ctx, config)
	if err != nil {
		return nil, err
	}
	return &typedResponsesAPIModel[M]{base: base}, nil
}

func (m *typedResponsesAPIModel[M]) Generate(ctx context.Context, input []M, opts ...model.Option) (M, error) {
	var zero M
	if isAgentic[M]() {
		msgInput := agenticMessagesToMessages(castAgenticSlice(input))
		out, err := m.base.Generate(ctx, msgInput, opts...)
		if err != nil {
			return zero, err
		}
		return any(messageToAgentic(out)).(M), nil
	}
	out, err := m.base.Generate(ctx, castMessageSlice(input), opts...)
	if err != nil {
		return zero, err
	}
	return any(out).(M), nil
}

func (m *typedResponsesAPIModel[M]) Stream(ctx context.Context, input []M, opts ...model.Option) (*schema.StreamReader[M], error) {
	if isAgentic[M]() {
		msgInput := agenticMessagesToMessages(castAgenticSlice(input))
		out, err := m.base.Stream(ctx, msgInput, opts...)
		if err != nil {
			return nil, err
		}
		r, w := schema.Pipe[M](1)
		go func() {
			defer w.Close()
			chunks := make([]*schema.Message, 0, 16)
			for {
				c, recvErr := out.Recv()
				if errors.Is(recvErr, io.EOF) {
					break
				}
				if recvErr != nil {
					_ = w.Send(*new(M), recvErr)
					return
				}
				chunks = append(chunks, c)
			}
			if len(chunks) == 0 {
				return
			}
			merged, mergeErr := schema.ConcatMessages(chunks)
			if mergeErr != nil {
				_ = w.Send(*new(M), mergeErr)
				return
			}
			_ = w.Send(any(messageToAgentic(merged)).(M), nil)
		}()
		return r, nil
	}

	out, err := m.base.Stream(ctx, castMessageSlice(input), opts...)
	if err != nil {
		return nil, err
	}
	return schema.StreamReaderWithConvert(out, func(in *schema.Message) (M, error) {
		return any(in).(M), nil
	}), nil
}

func isAgentic[M messageType]() bool {
	var zero M
	_, ok := any(zero).(*schema.AgenticMessage)
	return ok
}

func castMessageSlice[M messageType](in []M) []*schema.Message {
	out := make([]*schema.Message, 0, len(in))
	for _, m := range in {
		out = append(out, any(m).(*schema.Message))
	}
	return out
}

func castAgenticSlice[M messageType](in []M) []*schema.AgenticMessage {
	out := make([]*schema.AgenticMessage, 0, len(in))
	for _, m := range in {
		out = append(out, any(m).(*schema.AgenticMessage))
	}
	return out
}

func agenticMessagesToMessages(in []*schema.AgenticMessage) []*schema.Message {
	out := make([]*schema.Message, 0, len(in))
	for _, am := range in {
		if am == nil {
			continue
		}

		toolResultHandled := false
		for _, b := range am.ContentBlocks {
			if b == nil || b.FunctionToolResult == nil {
				continue
			}
			toolResultHandled = true
			out = append(out, schema.ToolMessage(functionToolResultToText(b.FunctionToolResult), b.FunctionToolResult.CallID))
		}
		if toolResultHandled {
			continue
		}

		msg := &schema.Message{Role: agenticRoleToRole(am.Role)}
		var textParts []string
		for _, b := range am.ContentBlocks {
			if b == nil {
				continue
			}
			if b.UserInputText != nil {
				textParts = append(textParts, b.UserInputText.Text)
			}
			if b.AssistantGenText != nil {
				textParts = append(textParts, b.AssistantGenText.Text)
			}
			if b.FunctionToolCall != nil {
				msg.ToolCalls = append(msg.ToolCalls, schema.ToolCall{
					ID: b.FunctionToolCall.CallID,
					Function: schema.FunctionCall{
						Name:      b.FunctionToolCall.Name,
						Arguments: b.FunctionToolCall.Arguments,
					},
				})
			}
		}
		msg.Content = strings.Join(textParts, "")
		if msg.Content != "" || len(msg.ToolCalls) > 0 {
			out = append(out, msg)
		}
	}
	return out
}

func messageToAgentic(msg *schema.Message) *schema.AgenticMessage {
	if msg == nil {
		return nil
	}
	am := &schema.AgenticMessage{Role: roleToAgenticRole(msg.Role)}
	if msg.Content != "" {
		if am.Role == schema.AgenticRoleTypeAssistant {
			am.ContentBlocks = append(am.ContentBlocks, schema.NewContentBlock(&schema.AssistantGenText{Text: msg.Content}))
		} else {
			am.ContentBlocks = append(am.ContentBlocks, schema.NewContentBlock(&schema.UserInputText{Text: msg.Content}))
		}
	}
	for _, tc := range msg.ToolCalls {
		am.Role = schema.AgenticRoleTypeAssistant
		am.ContentBlocks = append(am.ContentBlocks, schema.NewContentBlock(&schema.FunctionToolCall{
			CallID:    tc.ID,
			Name:      tc.Function.Name,
			Arguments: tc.Function.Arguments,
		}))
	}
	if msg.Role == schema.Tool {
		am.Role = schema.AgenticRoleTypeUser
		am.ContentBlocks = []*schema.ContentBlock{schema.NewContentBlock(&schema.FunctionToolResult{
			CallID: msg.ToolCallID,
			Name:   "tool_result",
			Content: []*schema.FunctionToolResultContentBlock{{
				Type: schema.FunctionToolResultContentBlockTypeText,
				Text: &schema.UserInputText{Text: msg.Content},
			}},
		})}
	}
	return am
}

func functionToolResultToText(fr *schema.FunctionToolResult) string {
	if fr == nil {
		return ""
	}
	parts := make([]string, 0, len(fr.Content))
	for _, c := range fr.Content {
		if c != nil && c.Text != nil {
			parts = append(parts, c.Text.Text)
		}
	}
	return strings.Join(parts, "\n")
}

func roleToAgenticRole(role schema.RoleType) schema.AgenticRoleType {
	switch role {
	case schema.Assistant:
		return schema.AgenticRoleTypeAssistant
	case schema.System:
		return schema.AgenticRoleTypeSystem
	default:
		return schema.AgenticRoleTypeUser
	}
}

func agenticRoleToRole(role schema.AgenticRoleType) schema.RoleType {
	switch role {
	case schema.AgenticRoleTypeAssistant:
		return schema.Assistant
	case schema.AgenticRoleTypeSystem:
		return schema.System
	default:
		return schema.User
	}
}
