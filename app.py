from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
import numpy as np  ### 🔹 CAMBIO
import math  ### 🔹 CAMBIO


load_dotenv(override=True)

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )


def record_user_details(email, name="Nombre no indicado", notes="no proporcionadas"):
    push(f"Registrando {name} con email {email} y notas {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Registrando {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Utiliza esta herramienta para registrar que un usuario está interesado en estar en contacto y proporcionó una dirección de correo electrónico.",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "La dirección de email del usuario"
            },
            "name": {
                "type": "string",
                "description": "El nombre del usuario, si se indica"
            }
            ,
            "notes": {
                "type": "string",
                "description": "¿Alguna información adicional sobre la conversación que valga la pena registrar para dar contexto?"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Utiliza siempre esta herramienta para registrar cualquier pregunta que no haya podido responder porque no se sabía la respuesta.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "La pregunta no sabe responderse"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


class Me:

    def __init__(self):
        self.openai = OpenAI()
        self.name = "Andres Bedoya"
        reader = PdfReader("me/linkedin.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

    # ---------------------------------------------------------------------
    # ----------------------- EVALUADOR INTEGRADO -------------------------
    # ---------------------------------------------------------------------

    def evaluate_with_llm(self, question, response_text, context_text=""):
        """
        Evalúa la respuesta del modelo con una rúbrica predefinida usando el propio LLM.
        """
        rubric = f"""
        Eres un evaluador experto. Evalúa la respuesta según esta rúbrica y devuelve SOLO un JSON:
        - score: entero 0-100
        - relevancia: 0-100
        - fidelidad: 0-100
        - completitud: 0-100
        - tono: 0-100
        - hallucination_risk: uno de ["bajo","medio","alto"]
        - comments: observaciones breves (máx 120 palabras)

        CONTEXTO:
        {context_text}

        PREGUNTA:
        {question}

        RESPUESTA:
        {response_text}
        """

        resp = self.openai.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "system", "content": rubric}]
        )
        text = resp.choices[0].message.content

        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {
                "score": 0,
                "relevancia": 0,
                "fidelidad": 0,
                "completitud": 0,
                "tono": 0,
                "hallucination_risk": "alto",
                "comments": "El evaluador no devolvió JSON válido.",
            }
        return parsed

    def embedding_similarity_check(self, response_text, context_text, threshold=0.7):
        """
        Evalúa la similitud semántica entre respuesta y contexto usando embeddings.
        """
        if not context_text.strip():
            return {"cosine": None, "flag_low_similarity": False}

        emb_resp = self.openai.embeddings.create(
            input=response_text, model="text-embedding-3-small"
        ).data[0].embedding
        emb_ctx = self.openai.embeddings.create(
            input=context_text, model="text-embedding-3-small"
        ).data[0].embedding

        a, b = np.array(emb_resp), np.array(emb_ctx)
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))
        return {"cosine": cos, "flag_low_similarity": cos < threshold}

    def evaluate_response(self, question, response_text, context_text=""):
        """
        Fusiona evaluación textual (LLM) y semántica (embeddings).
        """
        eval_llm = self.evaluate_with_llm(question, response_text, context_text)
        emb_check = self.embedding_similarity_check(response_text, context_text)
        if emb_check["flag_low_similarity"]:
            eval_llm["fidelidad"] = max(0, eval_llm.get("fidelidad", 50) - 20)
            eval_llm["hallucination_risk"] = "alto"
            eval_llm["comments"] += " Baja similitud semántica con el contexto."
        eval_llm["_embedding_check"] = emb_check
        if "score" not in eval_llm:
            subs = [eval_llm.get(k, 50) for k in ("relevancia", "fidelidad", "completitud", "tono")]
            eval_llm["score"] = int(sum(subs) / len(subs))
        return eval_llm


    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self):
        system_prompt = f"""Actúas como {self.name}. Respondes preguntas en el sitio web de {self.name}, en particular preguntas relacionadas con la trayectoria profesional, los antecedentes, las habilidades y la experiencia de {self.name}.
            Tu responsabilidad es representar a {self.name} en las interacciones del sitio web con la mayor fidelidad posible.
            Se te proporciona un resumen de la trayectoria profesional y el perfil de LinkedIn de {self.name} que puedes usar para responder preguntas.
            Muestra un tono profesional y atractivo, como si hablaras con un cliente potencial o un futuro empleador que haya visitado el sitio web.
            Si no sabes la respuesta a alguna pregunta, usa la herramienta 'record_unknown_question' para registrar la pregunta que no pudiste responder, incluso si se trata de algo trivial o no relacionado con tu trayectoria profesional.
            Si el usuario participa en una conversación, intenta que se ponga en contacto por correo electrónico; pídele su correo electrónico y regístralo con la herramienta 'record_user_details'."""
        
        system_prompt += f"\n\n## Resumen:\n{self.summary}\n\n## Perfil de LinkedIn:\n{self.linkedin}\n\n"
        system_prompt += f"En este contexto, por favor chatea con el usuario, manteniéndote siempre en el personaje de {self.name}."
        return system_prompt
    
    def _convert_history_to_messages(self, history):
        """
        Convierte el formato de Gradio (tuplas) al formato de OpenAI (diccionarios).
        history: [[user_msg, bot_msg], ...]
        returns: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """
        messages = []
        for user_msg, bot_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if bot_msg:
                messages.append({"role": "assistant", "content": bot_msg})
        return messages
    
    def chat(self, message, history, evaluate=False):
        # Convertir history de formato Gradio a formato OpenAI
        history_messages = self._convert_history_to_messages(history)
        messages = [{"role": "system", "content": self.system_prompt()}] + history_messages + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        content = response.choices[0].message.content

        if evaluate:
            context_text = self.summary + "\n" + self.linkedin
            eval_result = self.evaluate_response(message, content, context_text=context_text)
            return content, eval_result
        return content
        

if __name__ == "__main__":
    me = Me()
    #gr.ChatInterface(me.chat, type="messages").launch()
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="gray")) as demo:
        gr.Markdown("# 🤖 Asistente de Andrés Bedoya")
        gr.Markdown("### Conoce mi experiencia profesional, proyectos y trayectoria en IA 💼")
        
        chatbot = gr.Chatbot(
            label="Chat profesional",
            height=500,
            show_copy_button=True,
            type="tuples",
            avatar_images=("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", None),
        )
        
        msg = gr.Textbox(
            placeholder="Escribe tu pregunta sobre Andrés...",
            label="💬 Tu mensaje"
        )
        
        clear = gr.Button("🧹 Limpiar chat")

        def respond(message, history):
            return "", history + [[message, me.chat(message, history)]]

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.launch()
