package a2ui

import (
	"errors"
	"fmt"
	"io"
	"log"
	"strings"

	"github.com/cloudwego/eino/adk"
	"github.com/cloudwego/eino/schema"
)

func StreamToWriterAgentic(w io.Writer, sessionID string, history []*schema.Message, events *adk.AsyncIterator[*adk.TypedAgentEvent[*schema.AgenticMessage]]) (string, string, int, error) {
	surfaceID := "chat-" + sessionID
	rootChildren := make([]string, 0, len(history))
	for i := range history {
		rootChildren = append(rootChildren, fmt.Sprintf("msg-%d-card", i))
	}
	if err := emit(w, Message{BeginRendering: &BeginRenderingMsg{SurfaceID: surfaceID, Root: "root-col"}}); err != nil {
		return "", "", 0, err
	}
	if err := emitHistory(w, surfaceID, history, rootChildren); err != nil {
		return "", "", 0, err
	}
	msgIdx := len(history)
	lastContent, interruptID, err := streamEventsAgentic(w, surfaceID, &rootChildren, &msgIdx, events)
	return lastContent, interruptID, msgIdx, err
}

func StreamContinueAgentic(w io.Writer, sessionID string, startMsgIdx int, events *adk.AsyncIterator[*adk.TypedAgentEvent[*schema.AgenticMessage]]) (string, string, int, error) {
	surfaceID := "chat-" + sessionID
	rootChildren := make([]string, startMsgIdx)
	for i := range rootChildren {
		rootChildren[i] = fmt.Sprintf("msg-%d-card", i)
	}
	msgIdx := startMsgIdx
	lastContent, interruptID, err := streamEventsAgentic(w, surfaceID, &rootChildren, &msgIdx, events)
	return lastContent, interruptID, msgIdx, err
}

func streamEventsAgentic(w io.Writer, surfaceID string, rootChildren *[]string, msgIdx *int, events *adk.AsyncIterator[*adk.TypedAgentEvent[*schema.AgenticMessage]]) (string, string, error) {
	var lastContent strings.Builder
	var interruptID string
	for {
		event, ok := events.Next()
		if !ok {
			log.Printf("[a2ui] event stream ended (iterator exhausted)")
			break
		}
		if event.Err != nil {
			_ = emitToolChip(w, surfaceID, rootChildren, msgIdx, "error", event.Err.Error())
			return lastContent.String(), "", event.Err
		}
		if event.Action != nil && event.Action.Interrupted != nil {
			ictxs := event.Action.Interrupted.InterruptContexts
			var desc string
			for _, ic := range ictxs {
				if ic.IsRootCause {
					interruptID = ic.ID
					desc = fmt.Sprintf("%v", ic.Info)
					break
				}
			}
			if interruptID == "" && len(ictxs) > 0 {
				interruptID = ictxs[0].ID
				desc = fmt.Sprintf("%v", ictxs[0].Info)
			}
			_ = emitToolChip(w, surfaceID, rootChildren, msgIdx, "approval needed", desc)
			_ = emit(w, Message{InterruptRequest: &InterruptRequestMsg{InterruptID: interruptID, Description: desc}})
			break
		}
		if event.Output == nil || event.Output.MessageOutput == nil {
			if event.Action != nil && event.Action.Exit {
				break
			}
			continue
		}

		mo := event.Output.MessageOutput
		msg, err := concatAgenticOutput(mo)
		if err != nil {
			_ = emitToolChip(w, surfaceID, rootChildren, msgIdx, "error", err.Error())
			return lastContent.String(), "", err
		}
		if msg == nil {
			continue
		}
		logAgenticStruct(msg)
		assistantText, toolCalls, toolResults := parseAgenticMessage(msg)
		for _, tc := range toolCalls {
			_ = emitToolChip(w, surfaceID, rootChildren, msgIdx, "tool call", formatToolCall(tc))
		}
		for _, tr := range toolResults {
			_ = emitToolChip(w, surfaceID, rootChildren, msgIdx, "tool result", tr)
		}
		if assistantText != "" {
			if err := emitTextCard(w, surfaceID, rootChildren, msgIdx, "Agent", assistantText); err != nil {
				return lastContent.String(), "", err
			}
			lastContent.Reset()
			lastContent.WriteString(assistantText)
		}

		if event.Action != nil && event.Action.Exit {
			break
		}
	}
	return lastContent.String(), interruptID, nil
}

func concatAgenticOutput(mo *adk.TypedMessageVariant[*schema.AgenticMessage]) (*schema.AgenticMessage, error) {
	if mo.IsStreaming && mo.MessageStream != nil {
		chunks := make([]*schema.AgenticMessage, 0, 16)
		for {
			c, err := mo.MessageStream.Recv()
			if errors.Is(err, io.EOF) {
				break
			}
			if err != nil {
				return nil, err
			}
			chunks = append(chunks, c)
		}
		if len(chunks) == 0 {
			return nil, nil
		}
		return schema.ConcatAgenticMessages(chunks)
	}
	return mo.Message, nil
}

func parseAgenticMessage(msg *schema.AgenticMessage) (string, []toolCallInfo, []string) {
	var text strings.Builder
	toolCalls := make([]toolCallInfo, 0)
	toolResults := make([]string, 0)
	for _, b := range msg.ContentBlocks {
		if b == nil {
			continue
		}
		if b.AssistantGenText != nil {
			text.WriteString(b.AssistantGenText.Text)
		}
		if b.UserInputText != nil && msg.Role == schema.AgenticRoleTypeAssistant {
			text.WriteString(b.UserInputText.Text)
		}
		if b.FunctionToolCall != nil {
			toolCalls = append(toolCalls, toolCallInfo{Name: b.FunctionToolCall.Name, Args: b.FunctionToolCall.Arguments})
		}
		if b.FunctionToolResult != nil {
			toolResults = append(toolResults, functionToolResultBlocksText(b.FunctionToolResult))
		}
	}
	return text.String(), toolCalls, toolResults
}

func functionToolResultBlocksText(fr *schema.FunctionToolResult) string {
	if fr == nil {
		return ""
	}
	parts := make([]string, 0, len(fr.Content))
	for _, cb := range fr.Content {
		if cb != nil && cb.Text != nil {
			parts = append(parts, cb.Text.Text)
		}
	}
	return strings.Join(parts, "\n")
}
