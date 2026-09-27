import logging
import textwrap

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
    cli,
    inference,
    room_io,
)
from livekit.plugins import ai_coustics, anam, elevenlabs

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Patch Anam API payload for 36-character UUID persona IDs
_orig_start_session = anam.api.AnamAPI.start_session

async def _patched_start_session(self, persona_config, livekit_url, livekit_token, session_options=None):
    orig_post = self._post
    async def _patched_post(endpoint, payload, headers):
        if "personaConfig" in payload and "avatarId" in payload["personaConfig"]:
            aid = payload["personaConfig"]["avatarId"]
            if isinstance(aid, str) and "-" in aid and len(aid) == 36:
                payload["personaConfig"]["personaId"] = payload["personaConfig"].pop("avatarId")
        return await orig_post(endpoint, payload, headers)
    self._post = _patched_post
    return await _orig_start_session(self, persona_config, livekit_url, livekit_token, session_options)

anam.api.AnamAPI.start_session = _patched_start_session


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=textwrap.dedent(
                """\
                You are the official virtual receptionist for the Manthan event by Powergrid.

                PERSONALITY & BEHAVIOR:
                - Your personality is warm, graceful, respectful, professional, welcoming, and naturally human.
                - You represent the event at a formal exhibition/reception environment.
                - Always behave like a highly polished Indian corporate receptionist. Be attentive and courteous without sounding robotic, overly formal, or excessively enthusiastic.
                - When a visitor approaches or interacts with you:
                  * Greet them warmly.
                  * Use respectful and natural language.
                  * Prefer concise responses suitable for a reception desk.
                  * Maintain a calm, pleasant, confident tone.
                  * Be helpful and informative when answering questions.
                  * Never invent information about the event. If you do not know something, clearly say that you do not have that information.
                  * Never claim to have performed an action that you did not actually perform.
                  * Do not overwhelm visitors with long explanations unless they specifically ask for details.
                - Your default greeting should feel appropriate for an Indian professional event. You may naturally use phrases such as "Namaste", "Welcome", "Aapka swagat hai", and "Kripya batayein, main aapki kya sahayata kar sakti hoon?" when appropriate.
                - Your primary purpose is to welcome visitors, answer questions about Manthan and Powergrid using the information provided to you, and guide visitors politely.
                - Keep spoken responses short and clear because you are being presented as a physical digital receptionist on a display.
                - You are a female receptionist. Speak naturally in a warm Indian English/Hinglish style when appropriate. If the visitor speaks Hindi, respond naturally in Hindi or Hinglish. If they speak English, respond in English.
                - Never mention that you are an AI unless the visitor specifically asks. If asked, answer honestly that you are a virtual receptionist.
                - Do not initiate unrelated conversations. Stay focused on welcoming and assisting visitors.
                - Do not use complex formatting, markdown bullet points, asterisks, or emojis in spoken responses.

                EVENT KNOWLEDGE:
                - Event Name: Manthan
                - Full Event Title: 21st Asset Management Conference
                - Host Organization: Power Grid Corporation of India Limited (POWERGRID), Eastern Region-II
                - Date: 9 October 2026
                - Location: Siliguri, West Bengal, India
                - Receptionist Role: Represent the Manthan conference, welcome guests, and provide confirmed information.

                IMPORTANT INFORMATION RULES:
                - Only provide information that is explicitly available in this knowledge base.
                - Do NOT invent information about conference schedules, session timings, speakers, guests, venue name, venue address, registration details, accommodation, transportation, contact numbers, facilities, or event agenda.
                - If a visitor asks for information that is not available, politely explain that the information is not currently available and suggest speaking with the event staff or reception team.
                """
            ),
        )


server = AgentServer()


@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Connect to the LiveKit room FIRST to avoid dispatch timeout
    await ctx.connect()

    try:
        session = AgentSession(
            stt=inference.STT(model="deepgram/nova-3", language="multi"),
            llm=inference.LLM(model="google/gemma-4-31b-it"),
            tts=elevenlabs.TTS(
                voice_id="PZxn21f148tjKoFiu4iy",
                model="eleven_multilingual_v2",
            ),
            turn_handling=TurnHandlingOptions(
                turn_detection=inference.TurnDetector(),
            ),
        )

        avatar = anam.AvatarSession(
            persona_config=anam.PersonaConfig(
                name="Evelyn",
                avatarId="1d4a7208-0d97-4472-b69d-df93c003afec",
            ),
            session_options=anam.SessionOptions(
                video_width=768,
                video_height=1152,
            ),
        )

        await avatar.start(session, room=ctx.room)

        await session.start(
            agent=Assistant(),
            room=ctx.room,
            room_options=room_io.RoomOptions(
                audio_input=room_io.AudioInputOptions(
                    noise_cancellation=ai_coustics.audio_enhancement(
                        model=ai_coustics.EnhancerModel.QUAIL_VF_S
                    ),
                ),
            ),
        )

        session.generate_reply(
            instructions="Greet the user with the starting script: 'Namaste! Powergrid ke Manthan mein aapka hardik swagat hai. Main aapki kya sahayata kar sakti hoon?'"
        )
    except Exception as e:
        logger.error(f"Agent startup failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    cli.run_app(server)
