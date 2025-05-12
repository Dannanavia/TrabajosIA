# ChatBot.py
import os
import io
import re
import wave
import spacy
import PyPDF2
import pyaudio
import keyboard
import docx2txt
import moviepy.editor as mp
from pytube import YouTube
from pydub import AudioSegment
from google.cloud import speech
from google.cloud import language_v1
from moviepy.editor import VideoFileClip
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nlp = spacy.load('es_dep_news_trf')


def down_video(link):
    try:
        video_url = link
        video = YouTube(video_url)
        video_stream = video.streams.get_highest_resolution()
        video_file_path = video_stream.download("./videos")
        if video_file_path:
            # Convertir el video a WAV
            wav_file_path = os.path.splitext(video_file_path)[0] + ".wav"
            clip = VideoFileClip(video_file_path)
            clip.audio.write_audiofile(wav_file_path)
            clip.close()
            os.remove(video_file_path)
            print(
                "\nEl video se ha descargado y convertido a WAV exitosamente.")
            return wav_file_path
        else:
            print(
                "\nHa ocurrido un error al descargar el video.")
            return None
    except Exception as e:
        print("Ha ocurrido un error: " + str(e))
        return None


def extract_text_from_docx(docx_path):
    try:
        # Extraer texto del archivo .docx
        text = docx2txt.process(docx_path)
        return text
    except Exception as e:
        print("Error al extraer texto del archivo .docx:", e)
        return None


def extract_text_from_pdf(pdf_path):
    try:
        # Abrir el archivo PDF
        with open(pdf_path, 'rb') as file:
            # Crear un objeto PdfReader
            pdf_reader = PyPDF2.PdfReader(file)
            # Inicializar una variable para almacenar el texto extraído
            text = ""
            # Iterar sobre cada página del PDF
            for page_num in range(len(pdf_reader.pages)):
                # Obtener el texto de la página actual
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
            return text
    except Exception as e:
        print("Error al extraer texto del archivo PDF:", e)
        return None


def extract_text_from_file(file_path):
    if file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    elif file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    else:
        print("Formato de archivo no compatible.")
        return None


def convert_to_wav(audio_path):
    sound = AudioSegment.from_file(audio_path)
    wav_path = audio_path.rsplit('.', 1)[0] + ".wav"
    if sound.channels > 1:
        sound = sound.set_channels(1)
    sound.export(wav_path, format="wav")
    return wav_path


def extract_audio_from_video(video_path, audio_format="wav"):
    video = mp.VideoFileClip(video_path)
    audio_path = video_path.split('.')[0] + f".{audio_format}"
    video.audio.write_audiofile(audio_path)
    return audio_path


def transcribe_large_audio(audio_path, language='es-ES', segment_length=59999):
    audio_path = convert_to_wav(audio_path)
    sound = AudioSegment.from_file(audio_path)
    client = speech.SpeechClient()
    total_duration = len(sound)
    full_transcription = []

    for start in range(0, total_duration, segment_length):
        end = min(start + segment_length, total_duration)
        segment = sound[start:end]
        print(f"\nTranscribiendo segmento de {start} a {end} ms")
        audio_data = io.BytesIO()
        segment.export(audio_data, format='wav')
        audio_data = audio_data.getvalue()

        audio = speech.RecognitionAudio(content=audio_data)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sound.frame_rate,
            language_code=language,
            enable_automatic_punctuation=True
        )

        try:
            response = client.recognize(config=config, audio=audio)
            transcript = ' '.join(
                [result.alternatives[0].transcript for result in response.results])
            full_transcription.append(transcript)
        except Exception as e:
            full_transcription.append(
                "[Error de transcripción: {}]".format(str(e)))

    return " ".join(full_transcription)


def preprocess_text(text):
    # Agrega un punto después de una palabra seguida de un espacio y una letra mayúscula, si no está precedido por un punto, exclamación o interrogación
    processed_text = re.sub(r'(?<![\.\!\?])\s+([A-ZÁÉÍÓÚÑ])', r'. \1', text)
    return processed_text


def google_tokenize_text(text_content):
    # Tokenización de texto en frases mediante la API de lenguaje natural de Google
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document(
        content=text_content, type_=language_v1.Document.Type.PLAIN_TEXT)
    response = client.analyze_syntax(document=document)
    sentences = [sentence.text.content for sentence in response.sentences]
    return sentences


def process_phrases_with_spacy(sentences):
    processed_sentences = []
    for sentence in sentences:
        doc = nlp(sentence)
        processed_sentences.append(' '.join(
            [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]))
    return processed_sentences


def find_best_response(question, sentences):
    question_doc = nlp(question)
    question_tokens = [
        token.lemma_ for token in question_doc if not token.is_stop and not token.is_punct]
    vectorizer = TfidfVectorizer()
    all_texts = [' '.join(question_tokens)] + sentences
    vectors = vectorizer.fit_transform(all_texts)
    cosine_similarities = cosine_similarity(
        vectors[0:1], vectors[1:]).flatten()
    response_index = cosine_similarities.argmax()
    return sentences[response_index]


def record_question():
    # Graba la entrada del micrófono hasta que se presione la tecla "espacio"
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1,
                    rate=16000, input=True, frames_per_buffer=1024)
    print("\nGrabando, presione la tecla 'espacio' para detener...")
    frames = []

    while True:
        data = stream.read(1024)
        frames.append(data)
        # Detiene la grabación si se presiona 'espacio'
        if keyboard.is_pressed('space'):
            print("Grabación detenida.")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open('temp.wav', 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()
    return 'temp.wav'


def transcribe_question(audio_file):
    # Usar Google Speech-to-Text para transcribir la pregunta grabada
    client = speech.SpeechClient()
    with open(audio_file, 'rb') as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='es-ES',
        enable_automatic_punctuation=True
    )

    response = client.recognize(config=config, audio=audio)
    if response.results:
        transcript = ' '.join(
            [result.alternatives[0].transcript for result in response.results])
        # Imprime la pregunta del usuario
        print("\nPregunta por voz:", transcript)
        return transcript
    else:
        print("No se pudo transcribir la pregunta.")
        return ""


def chat_interface(sentences):
    processed_sentences = process_phrases_with_spacy(sentences)
    print("ChatBot activado con spaCy. Puede hacer una pregunta por voz o escribir su pregunta.")
    while True:
        choice = input(
            "\nPresione la tecla 'Enter' y hable para hacer una pregunta por voz, escriba su pregunta, o 'salir' para volver al menú: ")
        if choice.lower() == 'salir':
            print("")
            print("Volviendo al menú principal...")
            return  # Volver al menú principal en lugar de romper el bucle
        elif choice.strip() == '':
            # El usuario presionó Enter, grabar pregunta por voz
            audio_path = record_question()
            question = transcribe_question(audio_path)
        else:
            # El usuario escribió una pregunta
            question = choice

        if not question.strip():
            print("No se detectó una pregunta válida. Intente de nuevo.")
            continue

        response = find_best_response(question, processed_sentences)
        print("Respuesta:", response)


def main():
    running = True
    while running:
        print("")
        print("----------------------------------------")
        print("Bienvenido al ChatBot. ¿Qué desea hacer?")
        print("----------------------------------------")
        print("")
        print("1- Procesar archivos de video o audio")
        print("2- Procesar documentos de word o pdf")
        print("3- Procesar videos de YouTube")
        print("4- Salir")
        print("")

        option = int(
            input("Ingrese el número de la opción que desea ejecutar: "))

        if option == 1:
            file_path = input(
                "\nIngrese la ruta del archivo de video o audio: ")
            if file_path.split('.')[-1] in ['mp4', 'mkv', 'avi']:
                audio_path = extract_audio_from_video(file_path)
            else:
                audio_path = file_path
            try:
                transcription = transcribe_large_audio(audio_path)
                print("\nTranscripción del audio:")
                print(transcription)
                preprocessed_transcription = preprocess_text(transcription)
                sentences = google_tokenize_text(preprocessed_transcription)
                print("\nSentencias o frases:")
                for i, sentence in enumerate(sentences, 1):
                    print(f"Frase {i}: {sentence}")
                # Procesar las frases con spaCy antes de pasarlas al chatbot
                processed_sentences = process_phrases_with_spacy(sentences)

                # Iniciar la interfaz del chatbot
                print("\nChatBot activado. Puede empezar a hacer preguntas.")
                chat_interface(processed_sentences)
            except Exception as e:
                print("No fue posible realizar la transcripción del audio:", str(e))
        if option == 2:
            file_path = input(
                "\nIngrese la ruta del archivo (puede ser un archivo .docx o .pdf): ")
            # Extraer texto del archivo
            transcription = extract_text_from_file(file_path)
            try:
                sentences = google_tokenize_text(transcription)
                print("\nSentencias o frases:")
                for i, sentence in enumerate(sentences, 1):
                    print(f"Frase {i}: {sentence}")
                # Procesar las frases con spaCy antes de pasarlas al chatbot
                processed_sentences = process_phrases_with_spacy(sentences)

                # Iniciar la interfaz del chatbot
                print("\nChatBot activado. Puede empezar a hacer preguntas.")
                chat_interface(processed_sentences)
            except Exception as e:
                print("No fue posible obtener el texto", str(e))
        if option == 3:
            link = input("\nIngrese el link de YouTube: ")
            file_path = down_video(link)
            print("\nVideo guardado en", str(file_path))
            if file_path.split('.')[-1] in ['mp4', 'mkv', 'avi']:
                audio_path = extract_audio_from_video(file_path)
            else:
                audio_path = file_path
            try:
                transcription = transcribe_large_audio(audio_path)
                print("\nTranscripción del audio:")
                print(transcription)
                preprocessed_transcription = preprocess_text(transcription)
                sentences = google_tokenize_text(preprocessed_transcription)
                print("\nSentencias o frases:")
                for i, sentence in enumerate(sentences, 1):
                    print(f"Frase {i}: {sentence}")
                # Procesar las frases con spaCy antes de pasarlas al chatbot
                processed_sentences = process_phrases_with_spacy(sentences)

                # Iniciar la interfaz del chatbot
                print("\nChatBot activado. Puede empezar a hacer preguntas.")
                chat_interface(processed_sentences)
            except Exception as e:
                print(
                    "No fue posible realizar la transcripción del link de youtube por:", str(e))
        if option == 4:
            print("")
            print("---------------------------------------------------")
            print("Cerrando el programa. ¡Gracias por usar el ChatBot!")
            print("---------------------------------------------------")
            running = False


if __name__ == "__main__":
    main()
