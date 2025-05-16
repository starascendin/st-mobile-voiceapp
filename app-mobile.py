import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv
from audiorecorder import audiorecorder # Import the new audio recorder component
import tempfile # To save audio temporarily

# --- Configuration ---
load_dotenv() # Load environment variables from .env file

# --- IMPORTANT FOR iOS & BROWSER COMPATIBILITY ---
# 1. HTTPS: This app MUST be served over HTTPS for microphone access in Safari and other modern browsers.
# 2. User Interaction: Microphone access is typically granted only after a direct user gesture (e.g., tap).
#    The audiorecorder component itself acts as the initiator.
# 3. Component Behavior: The audiorecorder component's behavior on iOS Safari is key.
#    If it doesn't correctly request/handle mic permissions or the audio stream on iOS,
#    Python-level changes alone cannot fix it.

# Ensure GROQ_API_KEY is set (from .env, Streamlit secrets, or sidebar input)
# Initialize API key from environment or secrets if available
if "GROQ_API_KEY" not in os.environ:
    if hasattr(st, 'secrets') and "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    # Else, it will be prompted in the sidebar

DEFAULT_MODEL = "llama3-8b-8192"
DEFAULT_WHISPER_MODEL = "whisper-large-v3"

def get_groq_client():
    """
    Initializes and returns a Groq client.
    Stops the app if the API key is not configured.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        st.error("🔴 GROQ_API_KEY not set. Please enter it in the sidebar. The app functionality will be limited until the key is provided and applied.")
        st.stop() # Stop execution if key is missing
    return Groq(api_key=api_key)

def transcribe_audio_with_groq(client, audio_bytes, model_name):
    """
    Transcribes the given audio bytes using Groq's Whisper API.
    """
    if not audio_bytes:
        st.warning("No audio data to transcribe.")
        return None

    tmp_audio_file_path = None # Initialize for robust cleanup
    try:
        # Create a temporary file to store the audio bytes
        # audiorecorder provides WAV bytes when exported with format="wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio_file:
            tmp_audio_file.write(audio_bytes)
            tmp_audio_file_path = tmp_audio_file.name

        # Transcribe the audio file
        with open(tmp_audio_file_path, "rb") as audio_file_to_transcribe:
            transcription_response = client.audio.transcriptions.create(
                file=(os.path.basename(tmp_audio_file_path), audio_file_to_transcribe.read()),
                model=model_name,
            )
        
        return transcription_response.text
    except Exception as e:
        st.error(f"Error transcribing audio with Groq: {e}")
        return None
    finally:
        # Ensure the temporary file is deleted
        if tmp_audio_file_path and os.path.exists(tmp_audio_file_path):
            try:
                os.remove(tmp_audio_file_path)
            except Exception as e_remove:
                st.warning(f"Could not remove temporary audio file: {e_remove}")


def get_llm_response(client, user_prompt, model_name, chat_history_for_api):
    """
    Gets a response from the Groq LLM API based on the user prompt and chat history.
    """
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
    st.caption("Chat with LLMs and transcribe your voice memos. Recording will be auto-transcribed upon stopping.")
    st.markdown("""
    **Important for iOS (iPhone/iPad) Users:**
    - Audio recording requires your browser (Safari) to have microphone permissions for this site.
    - This app **must be accessed via HTTPS** (secure connection, URL starts with `https://`).
    - If recording doesn't start or yields no audio, it might be due to iOS Safari's specific media handling. Try on a desktop browser to verify core functionality.
    """)

    # Model selection defaults (can be overridden by sidebar)
    selected_llm_model = DEFAULT_MODEL
    selected_transcription_model = DEFAULT_WHISPER_MODEL

    # --- Sidebar for Configuration ---
    with st.sidebar:
        st.subheader("⚙️ Configuration")
        
        current_api_key = os.environ.get("GROQ_API_KEY", "")
        groq_api_key_input = st.text_input(
            "Enter Groq API Key:",
            type="password",
            value=current_api_key,
            key="api_key_input_sidebar",
            help="Your Groq API key is required for chat and transcription."
        )
        if groq_api_key_input != current_api_key and groq_api_key_input:
            os.environ["GROQ_API_KEY"] = groq_api_key_input
            st.success("✅ API Key updated for this session.")
        elif not os.environ.get("GROQ_API_KEY"):
             st.warning("⚠️ Please enter your Groq API Key to enable app features.")

        if os.environ.get("GROQ_API_KEY"):
            st.success("Groq API Key is set.")
        else:
            st.error("🔴 Groq API Key is NOT set. Transcription/Chat will fail.")
            
        st.markdown("Get your Groq API key from [console.groq.com](https://console.groq.com/keys)")
        st.markdown("---")

        st.subheader("LLM Configuration")
        available_llm_models = ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it", "gemma2-9b-it"]
        try:
            default_llm_index = available_llm_models.index(DEFAULT_MODEL)
        except ValueError:
            default_llm_index = 0
        selected_llm_model = st.selectbox(
            "Choose a Chat Model:",
            available_llm_models,
            index=default_llm_index,
            key="llm_model_select"
        )

        st.markdown("---")
        st.subheader("Transcription Configuration")
        available_transcription_models = ["whisper-large-v3"]
        selected_transcription_model = st.selectbox(
            "Choose a Transcription Model:",
            available_transcription_models,
            index=0,
            key="transcription_model_select"
        )
        st.markdown("---")
        st.markdown("Built with [Streamlit](https://streamlit.io) & [Groq](https://groq.com/)")
        st.markdown("Audio Recorder by [audiorecorder](https://github.com/Joooohan/streamlit-audiorecorder)") # Updated link

    client = get_groq_client()

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! I'm your Groq-powered assistant. Record a voice memo or type your message."}]
    if "current_chat_message" not in st.session_state:
        st.session_state.current_chat_message = ""
    if "staged_chat_message" not in st.session_state:
        st.session_state.staged_chat_message = None
    if "last_transcribed_audio_signature" not in st.session_state:
        st.session_state.last_transcribed_audio_signature = None
    if "recorded_audio_bytes" not in st.session_state:
        st.session_state.recorded_audio_bytes = None

    if st.session_state.staged_chat_message is not None:
        st.session_state.current_chat_message = st.session_state.staged_chat_message
        st.session_state.staged_chat_message = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.markdown("---")
    st.subheader("🎤 Record your message (auto-transcribes on stop):")
    
    # --- Audio Recorder using audiorecorder ---
    audio_segment_from_recorder = audiorecorder(
        "Click to record", 
        "Click to stop recording", 
        key="audio_recorder_main"
    )

    if len(audio_segment_from_recorder) > 0:
        # A new audio recording has been made and stopped
        # Export to WAV bytes
        try:
            audio_bytes = audio_segment_from_recorder.export(format="wav").read()
            current_audio_signature = len(audio_bytes) # Use length of bytes as signature

            if current_audio_signature > 0 and current_audio_signature != st.session_state.get("last_transcribed_audio_signature"):
                st.session_state.recorded_audio_bytes = audio_bytes
                st.info("New audio recorded. Transcribing automatically...")
                st.audio(st.session_state.recorded_audio_bytes, format="audio/wav") # Play back

                with st.spinner("Transcribing audio... please wait."):
                    transcribed_text = transcribe_audio_with_groq(
                        client,
                        st.session_state.recorded_audio_bytes,
                        selected_transcription_model
                    )

                if transcribed_text is not None:
                    st.session_state.staged_chat_message = transcribed_text
                    if transcribed_text.strip():
                        st.success("✅ Auto-transcription complete! Text added to input box below.")
                    else:
                        st.info("ℹ️ Transcription resulted in empty text (e.g., no speech detected or silence).")
                    st.session_state.last_transcribed_audio_signature = current_audio_signature
                else:
                    st.error("⚠️ Auto-transcription failed. Please check for errors or try again.")
                
                st.rerun()
            # If the audio signature is the same as the last one, do nothing to avoid re-processing
            # Or if current_audio_signature is 0 (should not happen if len(audio_segment_from_recorder) > 0)

        except Exception as e_export:
            st.error(f"Error processing recorded audio: {e_export}")
            # Potentially reset audio states if export fails
            st.session_state.recorded_audio_bytes = None
            st.session_state.last_transcribed_audio_signature = None

    elif st.session_state.get("last_transcribed_audio_signature") is not None and len(audio_segment_from_recorder) == 0:
        # This case handles if the component is re-rendered without new audio after a previous recording
        # Or if a user somehow clears the audiorecorder state without a new recording.
        # We can optionally reset the signature to allow a new "empty" recording to be processed if needed,
        # but for now, let's assume if len is 0, no new audio to process.
        pass


    st.markdown("---")
    st.text_area(
        "Your message to the chatbot:",
        key="current_chat_message",
        placeholder="Your transcribed audio will appear here, or type your message directly.",
        height=100
    )

    if st.button("Send to Chatbot", key="send_button_main", type="primary"):
        prompt_to_send = st.session_state.current_chat_message 
        if prompt_to_send and prompt_to_send.strip():
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
            
            st.session_state.staged_chat_message = ""
            st.session_state.last_transcribed_audio_signature = None
            st.session_state.recorded_audio_bytes = None
            st.rerun()
        else:
            st.warning("Please enter a message or record audio first.")

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

            st.session_state.staged_chat_message = "" 
            st.session_state.last_transcribed_audio_signature = None
            st.session_state.recorded_audio_bytes = None
            st.rerun()

    if len(st.session_state.messages) > 1:
        if st.button("Clear Chat History", key="clear_chat_button"):
            st.session_state.messages = [{"role": "assistant", "content": "Chat history cleared. How can I assist you now?"}]
            st.session_state.staged_chat_message = ""
            st.session_state.recorded_audio_bytes = None
            st.session_state.last_transcribed_audio_signature = None
            st.success("Chat history has been cleared.")
            st.rerun()

if __name__ == "__main__":
    main()