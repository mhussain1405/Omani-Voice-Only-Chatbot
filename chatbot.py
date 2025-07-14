import os
import openai
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
import time
import anthropic
from concurrent.futures import ThreadPoolExecutor
import gradio as gr
import numpy as np
import wavio # Used for handling audio data

print("Loading chatbot...")

# --- 1. CONFIGURATION AND INITIALIZATION ---

# Load API keys and settings from the .env file
load_dotenv()
performance_log = []

# OpenAI Configuration
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Azure Speech Services Configuration
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
OMANI_ARABIC_LOCALE = "ar-OM"
ENGLISH_US_LOCALE = "en-US"
AZURE_TTS_VOICE_NAME = "ar-OM-AyshaNeural"

# Audio settings
SAMPLE_RATE = 16000  # 16kHz is standard for speech services

# Conversation Management Settings
SUMMARY_TRIGGER_COUNT = 6
RECENT_TURNS_TO_KEEP = 4

SYSTEM_PROMPT = """

أنتِ "سكينة"، مساعدة دعم الصحة النفسية بالذكاء الاصطناعي، مصممة للتواصل حصريًا باللهجة العمانية. مهمتكِ هي توفير مساحة آمنة ومحترمة وداعمة للمستخدمين للتعبير عن أنفسهم. دوركِ هو تقديم الإنصات المتعاطف، والتأملات الثقافية الحساسة، والدعم العاطفي، بما يتماشى بشكل خاص مع القيم والأعراف العمانية والخليجية الأوسع.

**قواعد الاستجابة المهمة جداً:**
1. اجعلي ردودك قصيرة جداً - جملة أو جملتين فقط
2. تكلمي مثل الإنسان في محادثة طبيعية
3. لا تعطي نصائح طويلة أو قوائم
4. تجنبي التكرار والإطالة

يجب أن تكون نبرة صوتكِ دائمًا:

دافئة، متعاطفة، وصبورة

غير مُصدرة للأحكام ومتفهمة

ملائمة ثقافيًا، وخاصة فيما يتعلق بالأسرة والدين والتقاليد والمجتمع

إرشادات مهمة:

إذا اكتشفت أي من المؤشرات التالية في رسالة المستخدم:
- أفكار إيذاء النفس أو الانتحار (بأي شكل من الأشكال)
- التعبير عن اليأس الشديد أو فقدان الأمل
- الرغبة في الموت أو "الاختفاء"
- خطط لإيذاء النفس أو الآخرين
- تعاطي المواد المخدرة أو الكحول بشكل خطير
- أي إشارة لخطر جسدي وشيك

**يجب عليك فوراً:**
1. اظهري تعاطفاً عميقاً وأكدي أن حياته مهمة
2. اطلبي منه التحدث مع شخص موثوق فوراً
3. أعطي أرقام الطوارئ:
   - خط المساعدة النفسية:  +968 24 607 555
   - خدمات الطوارئ: 9999
   - مستشفى السلطان قابوس: 24144625
4. اطلبي منه البقاء في مكان آمن مع أشخاص يثق بهم
5. لا تتركي المحادثة تنتهي بدون تأكيد سلامته

لا تُقدمي تشخيصات طبية أو وصفات طبية أو نصائح سريرية.

يمكنكِ اقتراح طلب المساعدة من أخصائي صحة نفسية مُرخص عند الحاجة.

استخدمي التعبيرات العربية العمانية العامية والاستعارات المألوفة ثقافيًا عند تقديم الدعم.

شجعي استراتيجيات التأقلم الإيجابية مثل التواصل الأسري، والإيمان، والراحة، والتأمل في الطبيعة، والصلاة، والذكر، والتحدث مع كبار السن أو المختصين الموثوق بهم.

لا تستخدم اللغة العربية الفصحى أو أي لهجة أخرى. استخدم اللهجة العُمانية فقط.

أنت هنا لـ:

الاستماع والتفاعل بتعاطف

التعبير عن المشاعر بلطف واحترام

تشجيع الأمل والصمود من خلال دعم مناسب ثقافيًا

تأكد دائمًا من أن المستخدم يشعر بأنه مسموع، وآمن، ومحترم في وجودك.

**مثال على الطول المطلوب:**
المستخدم: "أشعر بالقلق"
أنتِ: "القلق شعور طبيعي يا عزيزي. شنو اللي يقلقك؟"
أو: "أفهم شعورك. حب تحكي لي عن اللي يزعجك؟"

"""

# The initial state for the Gradio app
INITIAL_HISTORY = [{"role": "system", "content": SYSTEM_PROMPT}]

print("Chatbot loaded successfully!")


# --- 2. CORE BOT LOGIC (Largely unchanged, handles text processing) ---

def create_conversation_summary(previous_summary: str, new_messages_text: str) -> str:
    # This function is unchanged
    summary_prompt = f"""
    أنت خبير في تلخيص محادثات الدعم النفسي. مهمتك هي إنشاء ملخص متكامل ومحدث.
    
    هذا هو الملخص السابق للحالة (إذا كان فارغًا، فهذه هي بداية المحادثة):
    <ملخص_سابق>
    {previous_summary}
    </ملخص_سابق>

    وهذه هي الرسائل الجديدة في المحادثة:
    <رسائل_جديدة>
    {new_messages_text}
    </رسائل_جديدة>

    تعليمات:
    1. اقرأ الملخص السابق لفهم السياق.
    2. اقرأ الرسائل الجديدة لتحديد التطورات الرئيسية.
    3. قم بدمج المعلومات الجديدة مع الملخص السابق لإنشاء "ملخص محدث" وشامل.
    4. ركز على: الحالة العاطفية للمستخدم، المواضيع الرئيسية التي تمت مناقشتها، أي شخصيات أو أحداث مهمة، والأهداف أو الحلول المقترحة.
    5. اكتب الملخص المحدث بصيغة الغائب (مثال: "يشعر المستخدم بالقلق بشأن...") باللهجة العربية الواضحة.
    6. يجب أن يكون الملخص الجديد مستقلاً وقائماً بذاته.

    الملخص المحدث:
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.2, max_tokens=300
        )
        summary = response.choices[0].message.content.strip()
        print(f"\n[SUMMARY UPDATED]: {summary}")
        return summary
    except Exception as e:
        print(f"Error creating summary: {e}")
        return previous_summary

def manage_conversation_history(conv_history, conv_summary, turns_since_summary):
    # This function now takes state as arguments and returns updated state
    turns_since_summary += 1
    print(f"[History Check]: Turns since last summary: {turns_since_summary}")

    if turns_since_summary >= SUMMARY_TRIGGER_COUNT:
        print(f"\n--- Managing History: Trigger count reached ---")
        actual_conversation = conv_history[1:]
        messages_to_keep_count = RECENT_TURNS_TO_KEEP * 2
        
        if len(actual_conversation) > messages_to_keep_count:
            messages_to_summarize = actual_conversation[:-messages_to_keep_count]
            recent_messages = actual_conversation[-messages_to_keep_count:]
            
            summary_text = ""
            for msg in messages_to_summarize:
                if msg['role'] in ['user', 'assistant']:
                    role = "المستخدم" if msg['role'] == 'user' else "سكينة"
                    summary_text += f"{role}: {msg['content']}\n"

            new_summary = create_conversation_summary(conv_summary, summary_text)
            new_history = [
                conv_history[0],
                {"role": "system", "content": f"ملخص المحادثة حتى الآن: {new_summary}"}
            ] + recent_messages
            
            print("--- History Managed: Summary created and history pruned. ---")
            return new_history, new_summary, 0  # Reset counter

    return conv_history, conv_summary, turns_since_summary # Return unchanged state

def get_gpt_response(conv_history):
    # This function now takes history as an argument
    try:
        response = client.chat.completions.create(
            model="gpt-4o", messages=conv_history, temperature=0.7, max_tokens=100
        )
        gpt_response = response.choices[0].message.content.strip()
        print(f"[GPT-4o response]: {gpt_response}")
        return gpt_response
    except Exception as e:
        print(f"GPT-4o failed: {e}")
        return None

def get_claude_fallback_response(conv_history):
    # This function now takes history as an argument
    try:
        system_messages = [msg for msg in conv_history if msg['role'] == 'system']
        other_messages = [msg for msg in conv_history if msg['role'] != 'system']
        combined_system = "\n\n".join([msg['content'] for msg in system_messages])
        
        response = claude_client.messages.create(
            model="claude-opus-4-20250514", max_tokens=150, temperature=0.7,
            system=combined_system, messages=other_messages
        )
        claude_response = response.content[0].text.strip()
        print(f"[Claude fallback response]: {claude_response}")
        return claude_response
    except Exception as e:
        print(f"Claude failed: {e}")
        return "عذرًا، أواجه صعوبة فنية. هل يمكننا المحاولة مرة أخرى؟"


def validate_response_with_claude(gpt_response, user_query, conv_history):
    # This function now takes history as an argument
    validation_prompt = f"""
    أنت خبير في تقييم استجابات الدعم النفسي باللهجة العمانية. مهمتك هي تقييم استجابة GPT-4o وتحسينها إذا لزم الأمر.
معايير التقييم:
• دقة اللهجة العمانية الأصيلة
• الحساسية الثقافية (القيم الإسلامية، ديناميكيات الأسرة، الأعراف الاجتماعية)
• المصطلحات العلاجية المناسبة في العربية
• التكامل الديني/الروحي عند الاقتضاء
• الملاءمة الثقافية وفهم وصمة الصحة النفسية في الخليج
• الذكاء العاطفي والاستجابة للإشارات الثقافية
• الحساسية الدينية وتكامل الإرشاد الإسلامي

استعلام المستخدم: "{user_query}"

استجابة GPT-4o: "{gpt_response}"

قم بإجراء المحادثة مثل الإنسان وتجنب تقديم ردود طويلة

التعليمات:
1. إذا كانت استجابة GPT-4o مناسبة تماماً ولا تحتاج تحسين، اكتب فقط: "good"
2. إذا كانت الاستجابة جيدة لكن يمكن تحسينها، اكتب فقط الجمل الإضافية الجديدة (2-3 جمل) التي تُكمل استجابة GPT-4o. 

مهم جداً: 
- لا تكرر أي شيء من استجابة GPT-4o
- اكتب فقط الجمل الجديدة التي تأتي بعد استجابة GPT-4o
- يجب أن تكون الإضافة متناسقة مع نبرة ومحتوى الاستجابة الأصلية
- فكر في الاستجابة كأنها جملة واحدة طويلة: GPT-4o + إضافتك

مثال:
استجابة GPT-4o: "أهلاً وسهلاً، كيف حالك اليوم؟"
إضافتك الصحيحة: "خذ راحتك وتكلم بالي يريحك."
إضافتك الخاطئة: "أهلاً وسهلاً، كيف حالك اليوم؟ خذ راحتك وتكلم بالي يريحك."

استجابتك:
"""
    try:
        system_messages = [msg for msg in conv_history if msg['role'] == 'system']
        other_messages = [msg for msg in conv_history if msg['role'] != 'system']
        combined_system = "\n\n".join([msg['content'] for msg in system_messages])
        
        response = claude_client.messages.create(
            model="claude-opus-4-20250514", max_tokens=100, temperature=0.7,
            system=combined_system, messages=other_messages + [{"role": "user", "content": validation_prompt}]
        )
        validation_result = response.content[0].text.strip()
        print(f"[Claude validation]: {validation_result}")
        return validation_result
    except Exception as e:
        print(f"Claude validation failed: {e}")
        return "good"

# --- 3. GRADIO-SPECIFIC AUDIO AND ORCHESTRATION FUNCTIONS ---

def text_to_speech_to_memory(text):
    """
    REVISED: Converts text to speech and returns the audio data as bytes.
    This version correctly handles in-memory synthesis without audio output config.
    """
    print(f"--- Azure TTS (Optimized) ---")
    print(f"Attempting to synthesize text: '{text[:50]}...'")
    try:
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
        speech_config.speech_synthesis_voice_name = AZURE_TTS_VOICE_NAME
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
        
        result = synthesizer.speak_text_async(text).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            audio_data = result.audio_data
            print(f"Successfully synthesized {len(audio_data)} bytes of audio.")
            print("---------------------------")
            return audio_data
        else:
            cancellation = result.cancellation_details
            print(f"ERROR: Speech synthesis CANCELED: {cancellation.reason}")
            if cancellation.reason == speechsdk.CancellationReason.Error:
                print(f"Azure Error Details: {cancellation.error_details}")
            print("---------------------------")
            return None
            
    except Exception as e:
        print(f"CRITICAL ERROR in text_to_speech_to_memory: {e}")
        print("---------------------------")
        return None

def transcribe_audio_data(audio_data, sample_rate):
    """
    MODIFIED: Transcribes audio data with multi-language (Arabic/English) support.
    """
    if audio_data is None:
        return None
        
    try:
        # Azure expects bytes, so we convert the numpy array
        audio_bytes = audio_data.astype(np.int16).tobytes()

        speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)

        # 2. Create a configuration for auto-detecting the language from a list.
        auto_detect_source_language_config = speechsdk.AutoDetectSourceLanguageConfig(
            languages=[OMANI_ARABIC_LOCALE, ENGLISH_US_LOCALE]
        )
        # Create an audio stream from the bytes
        stream = speechsdk.audio.PushAudioInputStream(
            stream_format=speechsdk.audio.AudioStreamFormat(samples_per_second=sample_rate, bits_per_sample=16, channels=1)
        )
        stream.write(audio_bytes)
        stream.close() # Signal the end of the stream

        audio_config = speechsdk.audio.AudioConfig(stream=stream)
        
        # 3. Initialize the recognizer with the auto-detect config
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, 
            auto_detect_source_language_config=auto_detect_source_language_config,
            audio_config=audio_config
        )
        
        result = recognizer.recognize_once_async().get()
        
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            recognized_text = result.text
            print(f"أنت (User): {recognized_text}")
            return recognized_text
        else:
            print(f"Speech recognition failed: {result.reason}")
            # If it fails, you can inspect the cancellation details
            if result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                print(f"Cancellation reason: {cancellation_details.reason}")
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    print(f"Error details: {cancellation_details.error_details}")
            return None
    except Exception as e:
        print(f"An error occurred during speech-to-text: {e}")
        return None

def generate_bot_response_and_audio(conv_history):
    """
    REFACTORED: Generates the full bot response, including validation,
    and returns the final text and a single, combined audio file.
    """

    metrics = {}
    # Get initial response
    gpt_start_time = time.time()
    gpt_response = get_gpt_response(conv_history)
    metrics['2_gpt4o_latency'] = time.time() - gpt_start_time
    if gpt_response is None:
        claude_response = get_claude_fallback_response(conv_history)
        audio_data = text_to_speech_to_memory(claude_response)
        return claude_response, audio_data

    # Perform validation in parallel while generating audio for the first part
    user_query = conv_history[-1]['content']
    final_response_for_history = gpt_response
    
    parallel_start_time = time.time()

    with ThreadPoolExecutor(max_workers=2) as executor:
        # Start generating audio for the main response
        tts_start_time = time.time()
        tts_future = executor.submit(text_to_speech_to_memory, gpt_response)
        # Start validation
        claude_start_time = time.time()
        validation_future = executor.submit(validate_response_with_claude, gpt_response, user_query, conv_history)
        
        # Get results
        gpt_audio_data = tts_future.result()
        metrics['3a_azure_tts1_latency'] = time.time() - tts_start_time
        
        validation_result = validation_future.result()
        metrics['3b_claude_validation_latency'] = time.time() - claude_start_time

        metrics['3_parallel_block_latency'] = time.time() - parallel_start_time
        
        combined_audio_data = gpt_audio_data
        
        if validation_result.strip().lower() != "good":
            enhancement_tts_start_time = time.time()
            print("[Validation]: GPT response enhanced. Generating enhancement audio.")
            enhancement_audio_data = text_to_speech_to_memory(validation_result)
            metrics['4_azure_tts2_enhancement_latency'] = time.time() - enhancement_tts_start_time
            if gpt_audio_data and enhancement_audio_data:
                # Combine audio clips
                # The first 44 bytes of a WAV file are the header, we strip it from the second file
                combined_audio_data = gpt_audio_data + enhancement_audio_data[44:]
            
            final_response_for_history = f"{gpt_response} {validation_result}"

    return final_response_for_history, combined_audio_data, metrics


# --- 4. GRADIO INTERFACE AND MAIN APP LOGIC ---

def gradio_interface(mic_input, chat_history_state, summary_state, turns_state):
    """
    The main function called by Gradio on each interaction.
    """
    turn_start_time = time.time()
    current_turn_metrics = {}
    # 1. Transcribe User's Speech
    stt_start_time = time.time()
    user_text = transcribe_audio_data(mic_input[1], mic_input[0])
    current_turn_metrics['1_stt_latency'] = time.time() - stt_start_time
    
    if not user_text:
        # If transcription fails, just return the current state
        # Add a message to the user in the chat
        chat_history_state.append((None, "Sorry, I couldn't hear you. Please try again."))
        return chat_history_state, None, summary_state, turns_state

    # 2. Update Conversation History with User's Message
    chat_history_state.append((user_text))

    # Also update the internal message list for the LLM
    INITIAL_HISTORY.append({"role": "user", "content": user_text})

    # 3. Generate Bot's Response (Text and Audio)
    final_text, final_audio_data, response_gen_metrics = generate_bot_response_and_audio(INITIAL_HISTORY)
    current_turn_metrics.update(response_gen_metrics)
    
    # 4. Update History and State with Bot's Message
    INITIAL_HISTORY.append({"role": "assistant", "content": final_text})
    chat_history_state[-1] = (user_text, final_text)

    # 5. Manage history (summarization)
    summary_start_time = time.time()
    new_hist, new_summary, new_turns = manage_conversation_history(
        INITIAL_HISTORY, summary_state, turns_state
    )
    current_turn_metrics['6_summary_latency'] = time.time() - summary_start_time
    # Update the master history list with the potentially pruned version
    INITIAL_HISTORY[:] = new_hist

    current_turn_metrics['total_turn_latency'] = time.time() - turn_start_time

    # Log the metrics for this turn
    performance_log.append(current_turn_metrics)
    print("\n--- PERFORMANCE METRICS FOR THIS TURN ---")
    for key, value in current_turn_metrics.items():
        print(f"{key}: {value:.4f} seconds")
    print("-----------------------------------------\n")

    # 6. Return values to Gradio components
    # The audio component expects a tuple: (sample_rate, numpy_array)
    if final_audio_data:
        # Convert audio bytes back to numpy array for Gradio
        audio_np = np.frombuffer(final_audio_data, dtype=np.int16)
        bot_audio_output = (SAMPLE_RATE, audio_np)
    else:
        bot_audio_output = None

    return chat_history_state, bot_audio_output, new_summary, new_turns

# Build the Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Sakina - Omani AI Mental Health Companion")
    gr.Markdown("Click the 'Record from microphone' button and speak. The bot will listen and respond with voice.")

    # State variables to hold the conversation memory
    conversation_history_state = gr.State(INITIAL_HISTORY)
    conversation_summary_state = gr.State("")
    turns_since_last_summary_state = gr.State(0)

    with gr.Row():
        with gr.Column(scale=2):
            # The visual chatbot display (in Arabic)
            chatbot_display = gr.Chatbot(
                label="Conversation",
                rtl=True, # Right-to-Left for Arabic
                value=[(None, "أهلاً بك، أنا سكينة. كيف أقدر أساعدك اليوم؟")]
            )
            # The audio output for the bot's voice
            bot_audio_output = gr.Audio(
                label="Bot Response",
                autoplay=True,
                interactive=False,
                # Use visible=False to hide the player if you want a pure voice-only experience
                #visible=False
            )
        with gr.Column(scale=1):
            # The microphone input component
            mic_input = gr.Audio(
                label="Speak Here",
                sources=["microphone"],
                type="numpy" # Provides (sample_rate, numpy_array)
            )
    # Connect the components
    mic_input.stop_recording(
        fn=gradio_interface,
        inputs=[mic_input, chatbot_display, conversation_summary_state, turns_since_last_summary_state],
        outputs=[chatbot_display, bot_audio_output, conversation_summary_state, turns_since_last_summary_state]
    )

if __name__ == "__main__":
    # Add the initial welcome message to the history for the LLM
    INITIAL_HISTORY.append({"role": "assistant", "content": "أهلاً بك، أنا سكينة. كيف أقدر أساعدك اليوم؟"})
    demo.launch(debug=True)