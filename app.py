import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv
from st_audiorec import st_audiorec # Import the audio recorder component
import tempfile # To save audio temporarily

# --- Configuration ---
load_dotenv()

# Ensure GROQ_API_KEY is set
if "GROQ_API_KEY" not in os.environ and hasattr(st, 'secrets') and "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

DEFAULT_MODEL = "llama3-8b-8192"
DEFAULT_WHISPER_MODEL = "whisper-large-v3"

def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not set. Please set it in the sidebar. The app will stop until the key is provided and applied.")
        st.stop()
    return Groq(api_key=api_key)

def transcribe_audio_with_groq(client, audio_bytes, model_name):
    if not audio_bytes:
        st.warning("No audio data to transcribe.")
        return None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
            tmp_audio_file.write(audio_bytes)
            tmp_audio_file_path = tmp_audio_file.name

        with open(tmp_audio_file_path, "rb") as audio_file_to_transcribe:
            transcription_response = client.audio.transcriptions.create(
                file=(os.path.basename(tmp_audio_file_path), audio_file_to_transcribe.read()),
                model=model_name,
            )
        os.remove(tmp_audio_file_path)
        return transcription_response.text
    except Exception as e:
        st.error(f"Error transcribing audio with Groq: {e}")
        if 'tmp_audio_file_path' in locals() and os.path.exists(tmp_audio_file_path):
            os.remove(tmp_audio_file_path)
        return None

def get_llm_response(client, user_prompt, model_name, chat_history_for_api):
    try:
        messages = chat_history_for_api + [{"role": "user", "content": user_prompt}]
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model_name,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error communicating with Groq LLM API: {e}")
        return None

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Groq Chat & Transcribe", page_icon="️🎙️", layout="centered")
    st.title("️🎙️ Groq Powered Chatbot with Voice Input")
    st.caption("Chat with LLMs and transcribe your voice memos. Recording will be auto-transcribed.")

    selected_llm_model = DEFAULT_MODEL # Defined from global scope
    selected_transcription_model = DEFAULT_WHISPER_MODEL # Defined from global scope

    with st.sidebar:
        st.subheader("Configuration")
        current_api_key = os.environ.get("GROQ_API_KEY", "")
        groq_api_key_input = st.text_input(
            "Enter Groq API Key:",
            type="password",
            value=current_api_key,
            key="api_key_input_sidebar"
        )
        if groq_api_key_input != current_api_key and groq_api_key_input:
            os.environ["GROQ_API_KEY"] = groq_api_key_input
            st.success("API Key updated for this session.")
            st.rerun()
        elif not current_api_key and not groq_api_key_input :
             st.warning("Please enter your Groq API Key.")

        if os.environ.get("GROQ_API_KEY"):
            st.success("Groq API Key is set.")
        else:
            st.error("Groq API Key is NOT set. Transcription/Chat will fail.")

        st.markdown("Get your Groq API key from [console.groq.com](https://console.groq.com/keys)")
        st.markdown("---")

        st.subheader("LLM Configuration")
        available_llm_models = ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it", "gemma2-9b-it"]
        try:
            default_llm_index = available_llm_models.index(DEFAULT_MODEL)
        except ValueError:
            default_llm_index = 0
        selected_llm_model = st.selectbox("Choose a Chat Model:", available_llm_models, index=default_llm_index, key="llm_model_select")

        st.markdown("---")
        st.subheader("Transcription Configuration")
        available_transcription_models = ["whisper-large-v3"]
        selected_transcription_model = st.selectbox("Choose a Transcription Model:", available_transcription_models, index=0, key="transcription_model_select")
        st.markdown("---")
        st.markdown("Built with [Streamlit](https://streamlit.io) & [Groq](https://groq.com/)")
        st.markdown("Audio Recorder by [st-audiorec](https://github.com/Joooohan/streamlit-audiorec)")

    client = get_groq_client()

    # Initialize session states
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm your Groq-powered assistant. Record a voice memo or type."}]
    if "current_chat_message" not in st.session_state:
        st.session_state.current_chat_message = ""
    if "staged_chat_message" not in st.session_state: # For staging changes to current_chat_message
        st.session_state.staged_chat_message = None
    if "last_transcribed_audio_signature" not in st.session_state:
        st.session_state.last_transcribed_audio_signature = None
    if "recorded_audio_bytes" not in st.session_state:
        st.session_state.recorded_audio_bytes = None

    # **Apply staged changes BEFORE the text_area is rendered**
    if st.session_state.staged_chat_message is not None:
        st.session_state.current_chat_message = st.session_state.staged_chat_message
        st.session_state.staged_chat_message = None # Reset the stage

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.markdown("---")
    st.subheader("🎤 Record your message (auto-transcribes on stop):")
    raw_audio_data_from_recorder = st_audiorec()

    if raw_audio_data_from_recorder is not None:
        current_audio_signature = len(raw_audio_data_from_recorder)
        if current_audio_signature > 0 and current_audio_signature != st.session_state.last_transcribed_audio_signature:
            st.session_state.recorded_audio_bytes = raw_audio_data_from_recorder
            st.info("New audio recorded. Transcribing automatically...")
            st.audio(st.session_state.recorded_audio_bytes, format="audio/wav")

            with st.spinner("Transcribing audio... please wait."):
                transcribed_text = transcribe_audio_with_groq(
                    client,
                    st.session_state.recorded_audio_bytes,
                    selected_transcription_model
                )

            if transcribed_text is not None:
                st.session_state.staged_chat_message = transcribed_text # Stage the update
                if transcribed_text:
                    st.success("Auto-transcription complete! Text added to input box below.")
                else:
                    st.info("Transcription resulted in empty text (e.g., no speech detected).")
                st.session_state.last_transcribed_audio_signature = current_audio_signature
            else:
                st.error("Auto-transcription failed. Please check console or try again.")
            st.rerun()

    st.markdown("---")
    st.text_area(
        "Your message to the chatbot:",
        key="current_chat_message",
        placeholder="Your transcribed audio will appear here, or type your message directly."
    )

    if st.button("Send to Chatbot", key="send_button"):
        prompt_to_send = st.session_state.current_chat_message # Read from the (now correctly populated) text_area state
        if prompt_to_send:
            st.session_state.messages.append({"role": "user", "content": prompt_to_send})
            with st.chat_message("user"):
                st.markdown(prompt_to_send)

            api_chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_llm_response(client, prompt_to_send, selected_llm_model, api_chat_history)
                if response:
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.error("Failed to get a response from the LLM.")
            
            st.session_state.staged_chat_message = "" # Stage the text area to be cleared
            st.session_state.last_transcribed_audio_signature = None
            st.session_state.recorded_audio_bytes = None
            st.rerun()
        else:
            st.warning("Please enter a message or record and transcribe audio first.")

    prompt_from_chat_input = st.chat_input("Or type your message here for quick send...", key="chat_input_main")
    if prompt_from_chat_input:
        st.session_state.messages.append({"role": "user", "content": prompt_from_chat_input})
        with st.chat_message("user"):
            st.markdown(prompt_from_chat_input)

        api_chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_llm_response(client, prompt_from_chat_input, selected_llm_model, api_chat_history)
            if response:
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.error("Failed to get a response from the LLM.")

            st.session_state.staged_chat_message = "" # Stage the main text area to be cleared too
            st.session_state.last_transcribed_audio_signature = None
            st.session_state.recorded_audio_bytes = None
            st.rerun()

    if len(st.session_state.messages) > 1:
        if st.button("Clear Chat History", key="clear_chat_button"):
            st.session_state.messages = [{"role": "assistant", "content": "Chat history cleared. How can I assist you now?"}]
            st.session_state.staged_chat_message = "" # Stage the text area to be cleared
            st.session_state.recorded_audio_bytes = None
            st.session_state.last_transcribed_audio_signature = None
            st.rerun()

if __name__ == "__main__":
    main()