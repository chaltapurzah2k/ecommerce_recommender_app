"""
Simple product chatbot agent using manual tool routing.
Uses Groq LLM + explicit tool calls without relying on function calling.
Optimized for low latency with single-shot LLM calls and efficient tool execution.
"""

import os
import re

from dotenv import load_dotenv

try:
    from langchain_groq import ChatGroq
    _CHATGROQ_IMPORT_ERROR = None
except Exception as exc:
    ChatGroq = None
    _CHATGROQ_IMPORT_ERROR = exc

from chatbot.tools import (
    filter_products_by_category,
    find_similar_products,
    get_product_details,
    get_top_popular_products,
    search_products,
    hybrid_search_products,
)

load_dotenv()

# Pre-compile regex patterns for reuse (eliminates recompilation overhead)
_TOOL_EXTRACT_PATTERN = re.compile(
    r"TOOL:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\n|\s+)PARAM:\s*(.+?)(?:\n|$)",
    re.IGNORECASE | re.DOTALL
)
_TOOL_CLEANUP_PATTERN = re.compile(
    r"TOOL:\s*[a-zA-Z_][a-zA-Z0-9_]*\s*(?:\n|\s+)PARAM:\s*[^\n]+",
    re.IGNORECASE
)
_PARAM_CLEANUP_PATTERN = re.compile(r"(?im)^\s*(TOOL|PARAM):\s*.*$")

_SYSTEM_PROMPT = """You are a helpful e-commerce shopping assistant for an online fashion and apparel store.

TOOL ROUTING RULES:
When the user asks a product question, respond with:
TOOL: <tool_name>
PARAM: <parameter_value>

THEN immediately provide a friendly response after the TOOL call (don't wait for results).

Available tools:
- hybrid_search_products: Advanced search with AI ranking (best for complex queries, typos, budget filters)
- search_products: Quick keyword search by name/category
- get_product_details: Detailed info about a specific product
- find_similar_products: "similar to X" or "more like X" recommendations
- get_top_popular_products: "what's trending", "popular", "most viewed"
- filter_products_by_category: Browse specific categories

IMPORTANT:
- PREFER hybrid_search_products for general searches (better relevance)
- Use search_products for simple, quick keyword lookups
- ALWAYS use a tool for product questions - never make up info
- Format: TOOL: <name>\nPARAM: <value>
- After TOOL call, write ONE friendly sentence (no need to wait for TOOL_RESULT)
- Keep responses under 3 sentences before tool output
- For non-product questions, respond naturally without a tool

Examples:
User: "show me running shoes under 5000"
Response: TOOL: hybrid_search_products
PARAM: running shoes under 5000
Let me find the best running shoes within your budget!

User: "what brands have boots"
Response: TOOL: search_products
PARAM: boots
I'll search for boots in our collection."""


class SimpleProductAgent:
    """Manual agent that routes to tools based on user intent. Optimized for single LLM call."""
    
    def __init__(self):
        if ChatGroq is None:
            raise ModuleNotFoundError(
                "langchain-groq is not installed. Install it with: pip install langchain-groq"
            ) from _CHATGROQ_IMPORT_ERROR

        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Add it to your .env file to enable the chatbot."
            )
        
        self.model = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            api_key=groq_key,
        )
        
        self.tools = {
            "search_products": search_products,
            "hybrid_search_products": hybrid_search_products,
            "get_product_details": get_product_details,
            "find_similar_products": find_similar_products,
            "get_top_popular_products": get_top_popular_products,
            "filter_products_by_category": filter_products_by_category,
        }

    def _clean_assistant_text(self, text: str) -> str:
        """Remove tool-routing artifacts from model responses."""
        if not text:
            return ""
        cleaned = _TOOL_CLEANUP_PATTERN.sub("", text)
        cleaned = re.sub(_PARAM_CLEANUP_PATTERN, "", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
    
    def _extract_tool_call(self, response_text: str) -> tuple[str, str] | None:
        """Extract tool name and parameter from model response (pre-compiled regex)."""
        match = _TOOL_EXTRACT_PATTERN.search(response_text)
        if match:
            tool_name = match.group(1).lower()
            param = match.group(2).strip()
            return tool_name, param
        return None
    
    def _call_tool(self, tool_name: str, param: str) -> str:
        """Execute the specified tool."""
        if tool_name not in self.tools:
            return f"Unknown tool: {tool_name}"
        
        tool_func = self.tools[tool_name]
        try:
            if tool_name == "get_top_popular_products":
                result = tool_func.invoke({"limit": 5})
            elif tool_name in ("search_products", "hybrid_search_products"):
                result = tool_func.invoke({"query": param})
            elif tool_name in ("get_product_details", "find_similar_products"):
                result = tool_func.invoke({"product_name": param})
            elif tool_name == "filter_products_by_category":
                result = tool_func.invoke({"category": param})
            else:
                result = tool_func.invoke({})
            return str(result)
        except Exception as e:
            return f"Error calling tool: {e}"
    
    def invoke(self, input_dict: dict, config: dict = None) -> dict:
        """
        Process user message and return agent response.
        OPTIMIZED: Single LLM call instead of two (no second call for formatting).
        """
        messages = input_dict.get("messages", [])
        if not messages:
            return {"messages": [{"role": "assistant", "content": "No message provided."}]}
        
        user_message = messages[-1].get("content", "")
        
        # Convert messages to format for ChatGroq
        chat_history = []
        chat_history.append({"role": "system", "content": _SYSTEM_PROMPT})
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            chat_history.append({"role": role, "content": content})
        
        # SINGLE LLM CALL: Get model's response with tool routing
        response = self.model.invoke(chat_history)
        response_text = response.content
        
        # Try to extract and execute tool call
        tool_call = self._extract_tool_call(response_text)
        
        if tool_call:
            tool_name, param = tool_call
            # Execute tool (non-blocking, result shown to user alongside model response)
            tool_result = self._call_tool(tool_name, param)
            
            # BUILD FINAL RESPONSE: Combine model response + tool result
            # No second LLM call needed - model already provided friendly intro
            cleaned_model_response = self._clean_assistant_text(response_text)
            
            # Format: model's friendly message + tool results
            final_lines = []
            if cleaned_model_response:
                final_lines.append(cleaned_model_response)
            final_lines.append(tool_result)
            final_text = "\n\n".join(final_lines)
        else:
            # No tool call detected, use model response directly
            final_text = self._clean_assistant_text(response_text)
            if not final_text:
                final_text = "I couldn't process that request. Try asking about products, categories, or what's trending."
        
        return {
            "messages": [
                {"role": "assistant", "content": final_text}
            ]
        }


class RuleBasedProductAgent:
    """Fallback agent used when Groq dependency/config is unavailable."""

    def __init__(self, reason: str = ""):
        self.reason = reason.strip()
        self.tools = {
            "search_products": search_products,
            "hybrid_search_products": hybrid_search_products,
            "get_product_details": get_product_details,
            "find_similar_products": find_similar_products,
            "get_top_popular_products": get_top_popular_products,
            "filter_products_by_category": filter_products_by_category,
        }

    def _call_tool(self, tool_name: str, param: str) -> str:
        tool_func = self.tools.get(tool_name)
        if tool_func is None:
            return "I couldn't decide the right tool. Please try another query."
        try:
            if tool_name == "get_top_popular_products":
                return str(tool_func.invoke({"limit": 5}))
            if tool_name in ("search_products", "hybrid_search_products"):
                return str(tool_func.invoke({"query": param}))
            if tool_name in ("get_product_details", "find_similar_products"):
                return str(tool_func.invoke({"product_name": param}))
            if tool_name == "filter_products_by_category":
                return str(tool_func.invoke({"category": param}))
            return str(tool_func.invoke({}))
        except Exception as exc:
            return f"Error while searching products: {exc}"

    def _route(self, user_text: str) -> tuple[str, str, str]:
        text = user_text.strip()
        lowered = text.lower()

        if any(token in lowered for token in ("trending", "popular", "most viewed", "top products")):
            return "get_top_popular_products", "", "Here are the products users interact with the most right now:"

        similar_match = re.search(r"(?:similar to|like)\s+(.+)$", text, flags=re.IGNORECASE)
        if similar_match:
            product_name = similar_match.group(1).strip()
            return "find_similar_products", product_name, f"Showing products similar to '{product_name}':"

        detail_match = re.search(r"(?:details of|price of|about)\s+(.+)$", text, flags=re.IGNORECASE)
        if detail_match:
            product_name = detail_match.group(1).strip()
            return "get_product_details", product_name, f"Here are details for '{product_name}':"

        category_match = re.search(
            r"\b(shoes|footwear|t-?shirts?|shirts?|jackets?|jeans|bags?|watches?|shorts|skirts?)\b",
            lowered,
        )
        if category_match and any(token in lowered for token in ("category", "show", "browse", "filter", "in ")):
            category = category_match.group(1)
            return "filter_products_by_category", category, f"Browsing products in '{category}':"

        return "hybrid_search_products", text, "I can still help with search while advanced chat is temporarily unavailable."

    def invoke(self, input_dict: dict, config: dict = None) -> dict:
        messages = input_dict.get("messages", [])
        if not messages:
            return {"messages": [{"role": "assistant", "content": "No message provided."}]}

        user_message = str(messages[-1].get("content", "")).strip()
        if not user_message:
            return {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "Please type a product query, such as 'running shoes under 5000' or 'similar to Nike jersey'.",
                    }
                ]
            }

        tool_name, param, preface = self._route(user_message)
        tool_output = self._call_tool(tool_name, param)

        lines = [preface]
        if self.reason:
            lines.append(f"(Fallback mode: {self.reason})")
        lines.append(tool_output)

        return {
            "messages": [
                {"role": "assistant", "content": "\n\n".join(lines)}
            ]
        }


def build_chatbot_agent():
    """Build and return the simple product agent. Cached by Streamlit."""
    if ChatGroq is None:
        reason = (
            "Missing dependency 'langchain-groq'. "
            "Install it in your active environment to enable the LLM assistant."
        )
        return RuleBasedProductAgent(reason=reason)

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        return RuleBasedProductAgent(
            reason="GROQ_API_KEY is not configured, so rule-based mode is active."
        )

    try:
        return SimpleProductAgent()
    except Exception as exc:
        return RuleBasedProductAgent(reason=str(exc))
