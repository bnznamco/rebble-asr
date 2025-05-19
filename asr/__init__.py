import io

import gevent.monkey

gevent.monkey.patch_all()
from email.mime.multipart import MIMEMultipart
from email.message import Message
import json
from speex import SpeexDecoder
import grpc.experimental.gevent as grpc_gevent
from groq import Groq
from groq.types.audio import Transcription

grpc_gevent.init_gevent()

from flask import Flask, request, Response, abort
import logging
import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
language = "it"

groq_client = Groq()


def transcribe_audio(chunks) -> Transcription:
    decoder = SpeexDecoder(1)
    pcm = bytearray()
    for chunk in chunks:
        pcm.extend(decoder.decode(chunk))
    try:
        return groq_client.audio.transcriptions.create(
            file=io.BytesIO(pcm),
            model="whisper-large-v3",
            language=language,
            response_format="json",
            temperature=0.0,
        )
    except Exception as e:
        logging.error("Error during transcription: %s", e)
        abort(500, "Transcription failed")


# We know gunicorn does this, but it doesn't *say* it does this, so we must signal it manually.
@app.before_request
def handle_chunking():
    request.environ["wsgi.input_terminated"] = 1


def parse_chunks(stream):
    boundary = (
        b"--"
        + request.headers["content-type"]
        .split(";")[1]
        .split("=")[1]
        .encode("utf-8")
        .strip()
    )  # super lazy/brittle parsing.
    this_frame = b""
    while True:
        content = stream.read(4096)
        this_frame += content
        end = this_frame.find(boundary)
        if end > -1:
            frame = this_frame[:end]
            this_frame = this_frame[end + len(boundary) :]
            if frame != b"":
                try:
                    header, content = frame.split(b"\r\n\r\n", 1)
                except ValueError:
                    continue
                yield content[:-2]
        if content == b"":
            break


@app.route("/heartbeat")
def heartbeat():
    return "asr"


@app.route("/NmspServlet/", methods=["POST"])
def recognise():
    stream = request.stream
    req_start = datetime.datetime.now()
    logging.info("Received transcription request at: %s", datetime.datetime.now())
    chunks = iter(list(parse_chunks(stream)))
    logging.info("Audio received in %s", datetime.datetime.now() - req_start)
    content = next(chunks).decode("utf-8")
    logging.info("Metadata: %s", content)

    asr_transcribe_start = datetime.datetime.now()
    trancription = transcribe_audio(chunks)

    logging.info(
        "ASR request completed in %s", datetime.datetime.now() - asr_transcribe_start
    )

    logging.info("Transcription: %s", trancription.text)

    words = []
    for result in trancription.text.split("\n"):
        words.extend(
            {
                "word": x,
                "confidence": 1.0,
            }
            for x in result.alternatives[0].transcript.split(" ")
        )

    # Now for some reason we also need to give back a mime/multipart message...
    parts = MIMEMultipart()
    response_part = Message()
    response_part.add_header("Content-Type", "application/JSON; charset=utf-8")

    if len(words) > 0:
        response_part.add_header("Content-Disposition", 'form-data; name="QueryResult"')
        words[0]["word"] += "\\*no-space-before"
        words[0]["word"] = words[0]["word"][0].upper() + words[0]["word"][1:]
        response_part.set_payload(json.dumps({"words": [words]}))
    else:
        response_part.add_header("Content-Disposition", 'form-data; name="QueryRetry"')
        # Other errors probably exist, but I don't know what they are.
        # This is a Nuance error verbatim.
        response_part.set_payload(
            json.dumps(
                {
                    "Cause": 1,
                    "Name": "AUDIO_INFO",
                    "Prompt": "Sorry, speech not recognized. Please try again.",
                }
            )
        )
    parts.attach(response_part)

    parts.set_boundary("--Nuance_NMSP_vutc5w1XobDdefsYG3wq")

    response = Response(
        "\r\n" + parts.as_string().split("\n", 3)[3].replace("\n", "\r\n")
    )
    response.headers["Content-Type"] = (
        f"multipart/form-data; boundary={parts.get_boundary()}"
    )
    logging.info("Request complete in %s", datetime.datetime.now() - req_start)
    return response
