import sys
import json
import asyncio
import gradio as gr
import ollama
from mcp_http_client_base import MCPHTTPClient


class MCPHTTPHostApp(MCPHTTPClient):
    """AI host application that uses a local Ollama model with MCP HTTP server tools."""

    def __init__(self, server_url: str, roots_dir: str):
        super().__init__(server_url, roots_dir)

        # Initialize local Ollama client
        self.llm_client = ollama.Client()
        self.model = "granite4:3b"

        # System prompt teaches the model the tool call format ollama parses
        self.conversation_history = [{
            "role": "system",
            "content": (
                "You are a helpful AI assistant with access to tools from an MCP server. "
                "To call a tool, output EXACTLY:\n"
                '<tool_call>{"name": "tool_name", "arguments": {"arg": value}}</tool_call>\n'
                "with nothing else in the message. Then wait for the tool results and answer the user."
            )
        }]

    async def get_available_tools(self):
        """Get all available tools in OpenAI function calling format."""
        await self.connect()

        # Get real MCP tools
        mcp_tools = await self.list_tools()

        openai_tools = []

        # Add real MCP tools
        for tool in mcp_tools:
            tool_schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or f"Execute {tool.name}",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }

            # Convert MCP input schema to OpenAI parameters
            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                schema = tool.inputSchema
                if isinstance(schema, dict):
                    if "properties" in schema:
                        tool_schema["function"]["parameters"]["properties"] = schema["properties"]
                    if "required" in schema and schema["required"]:
                        tool_schema["function"]["parameters"]["required"] = schema["required"]

            openai_tools.append(tool_schema)

        # Add synthetic tools for resources
        openai_tools.append({
            "type": "function",
            "function": {
                "name": "mcp_list_resources",
                "description": "List all available resources from the MCP server",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        })

        openai_tools.append({
            "type": "function",
            "function": {
                "name": "mcp_read_resource",
                "description": "Read a specific resource by URI from the MCP server",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "uri": {
                            "type": "string",
                            "description": "The URI of the resource to read (for example, 'file://workspace/example.txt')"
                        }
                    },
                    "required": ["uri"]
                }
            }
        })

        # Add synthetic tools for prompts
        openai_tools.append({
            "type": "function",
            "function": {
                "name": "mcp_list_prompts",
                "description": "List all available prompt templates from the MCP server",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        })

        openai_tools.append({
            "type": "function",
            "function": {
                "name": "mcp_get_prompt",
                "description": "Get a rendered prompt template from the MCP server",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the prompt template"
                        },
                        "arguments": {
                            "type": "object",
                            "description": "Arguments for the prompt template"
                        }
                    },
                    "required": ["name"]
                }
            }
        })

        return openai_tools

    async def execute_tool(self, tool_name: str, arguments: dict):
        """Execute a tool call (real MCP tool or synthetic helper)."""
        await self.connect()

        # Handle synthetic resource tools
        if tool_name == "mcp_list_resources":
            resources = await self.list_resources()
            result = "Available resources:\n"
            for resource in resources:
                result += f"- {resource.uriTemplate}"
                if resource.name:
                    result += f" ({resource.name})"
                if resource.description:
                    result += f": {resource.description}"
                result += "\n"
            return result

        if tool_name == "mcp_read_resource":
            uri = arguments.get("uri")
            if not uri:
                return "Error: URI is required"
            try:
                contents = await self.read_resource(uri)
                if isinstance(contents, list) and len(contents) > 0:
                    content = contents[0]
                    if hasattr(content, 'text'):
                        return content.text
                    return str(content)
                return str(contents)
            except Exception as e:
                return f"Error reading resource: {str(e)}"

        # Handle synthetic prompt tools
        if tool_name == "mcp_list_prompts":
            prompts = await self.list_prompts()
            result = "Available prompts:\n"
            for prompt in prompts:
                result += f"- {prompt.name}"
                if prompt.description:
                    result += f": {prompt.description}"
                if hasattr(prompt, 'arguments') and prompt.arguments:
                    args = [arg.name for arg in prompt.arguments]
                    result += f" (args: {', '.join(args)})"
                result += "\n"
            return result

        if tool_name == "mcp_get_prompt":
            name = arguments.get("name")
            prompt_args = arguments.get("arguments", {})
            if not name:
                return "Error: Prompt name is required"
            try:
                messages = await self.get_prompt(name, prompt_args)
                result = f"Prompt: {name}\n\n"
                for msg in messages:
                    role = getattr(msg, 'role', 'unknown')
                    content = getattr(msg, 'content', '')
                    if hasattr(content, 'text'):
                        content = content.text
                    result += f"[{role}]: {content}\n\n"
                return result
            except Exception as e:
                return f"Error getting prompt: {str(e)}"

        # Handle regular MCP tools
        try:
            result = await self.call_tool(tool_name, arguments)

            # Extract text content from result
            if isinstance(result, list) and len(result) > 0:
                content = result[0]
                if hasattr(content, 'text'):
                    text_result = content.text
                else:
                    text_result = str(content)
            elif hasattr(result, 'text'):
                text_result = result.text
            else:
                text_result = str(result)

            return text_result

        except Exception as e:
            return f"Error executing tool: {str(e)}"

    async def chat(self, user_message: str, history: list):
        """Chat with the local Ollama model using MCP tools."""
        await self.connect()

        # Add user message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Get available tools
        tools = await self.get_available_tools()

        # Agentic loop - continue while the model requests tool calls
        while True:
            response = await asyncio.to_thread(
                self.llm_client.chat,
                model=self.model,
                messages=self.conversation_history,
                tools=tools or None,
            )

            message = response["message"]

            # No tool calls - return final answer
            if not message.get("tool_calls"):
                final_message = message.get("content") or ""
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_message
                })
                return final_message

            # Add assistant's message with tool calls to history
            self.conversation_history.append({
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": message["tool_calls"]
            })

            # Execute each tool call
            for tool_call in message["tool_calls"]:
                function_name = tool_call["function"]["name"]
                function_args = tool_call["function"]["arguments"]
                if isinstance(function_args, str):
                    try:
                        function_args = json.loads(function_args) if function_args.strip() else {}
                    except json.JSONDecodeError:
                        function_args = {}

                # Execute the tool
                tool_result = await self.execute_tool(function_name, function_args)

                # Add tool result to history
                self.conversation_history.append({
                    "role": "tool",
                    "content": str(tool_result)
                })

    def create_interface(self):
        """Create the Gradio chat interface."""

        async def chat_wrapper(message, history):
            """Wrapper for chat method compatible with Gradio."""
            if not message.strip():
                return history

            response = await self.chat(message, history)
            # Return updated history with new messages
            return history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response}
            ]

        async def reset_conversation():
            """Reset the conversation history."""
            self.conversation_history = [self.conversation_history[0]]
            return []

        with gr.Blocks(title="MCP HTTP AI Host") as interface:
            gr.Markdown(f"""
            # MCP HTTP AI Host
            Chat with local Ollama gemma3-4k using tools from the MCP HTTP server.

            **Server:** {self.server_url}
            **Workspace Roots:** {self.roots_dir}
            **Model:** {self.model}

            The AI can use all available MCP tools, resources, and prompts during the conversation.
            """)

            chatbot = gr.Chatbot(
                label="Conversation",
                height=500
            )

            with gr.Row():
                msg = gr.Textbox(
                    label="Your message",
                    placeholder="Ask me to use MCP tools...",
                    scale=4
                )
                clear = gr.Button("Clear", scale=1)

            msg.submit(
                fn=chat_wrapper,
                inputs=[msg, chatbot],
                outputs=chatbot
            ).then(
                lambda: "",
                outputs=msg
            )

            clear.click(
                fn=reset_conversation,
                outputs=chatbot
            )

        return interface


def main():
    if len(sys.argv) < 3:
        print("Usage: python mcp_http_host_app.py <server_url> <roots_dir>")
        print("Example: python mcp_http_host_app.py http://127.0.0.1:8000 /path/to/workspace")
        sys.exit(1)

    server_url = sys.argv[1]
    roots_dir = sys.argv[2]

    client = MCPHTTPHostApp(server_url, roots_dir)
    interface = client.create_interface()
    interface.queue().launch(server_name="127.0.0.1", server_port=7862)


if __name__ == "__main__":
    main()
