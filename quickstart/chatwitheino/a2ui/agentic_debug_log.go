package a2ui

import (
	"encoding/json"
	"log"
	"os"
	"strings"

	"github.com/cloudwego/eino/schema"
)

func logAgenticStruct(msg *schema.AgenticMessage) {
	if msg == nil {
		return
	}
	v := strings.TrimSpace(os.Getenv("DEBUG_AGENTIC_STRUCT"))
	if v != "" && !(strings.EqualFold(v, "1") || strings.EqualFold(v, "true") || strings.EqualFold(v, "yes")) {
		return
	}
	b, err := json.Marshal(msg)
	if err != nil {
		log.Printf("[agentic-struct] marshal error: %v", err)
		return
	}
	log.Printf("[agentic-struct] %s", string(b))
}
