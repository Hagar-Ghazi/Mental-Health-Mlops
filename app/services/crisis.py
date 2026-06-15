from datetime import datetime, timezone
from sqlalchemy.orm import Session
from app.models import CrisisEvent

CRISIS_HOTLINES = {
    "Egypt": ("08008880700", "Egypt Crisis Line", "https://www.befrienders.org"),
    "Saudi Arabia": (
        "920033360",
        "KSA Mental Health Line",
        "https://www.befrienders.org",
    ),
    "United Arab Emirates": ("800HOPE", "UAE Hope Line", "https://www.befrienders.org"),
    "Jordan": ("110", "Jordan Lifeline", "https://www.befrienders.org"),
    "Lebanon": ("1564", "Embrace Lebanon", "https://embracelebanon.org"),
    "Morocco": ("0801004747", "Morocco Crisis Line", "https://www.befrienders.org"),
    "Tunisia": ("71108108", "Tunisia Mental Health", "https://www.befrienders.org"),
    "Kuwait": ("94006283", "Kuwait Crisis Support", "https://www.befrienders.org"),
    "Iraq": ("103", "Iraq Emergency", "https://www.befrienders.org"),
    "Libya": ("+218914590805", "Libya Crisis Support", "https://www.befrienders.org"),
    "United States": (
        "988",
        "988 Suicide & Crisis Lifeline",
        "https://988lifeline.org",
    ),
    "United Kingdom": ("116 123", "Samaritans", "https://www.samaritans.org"),
    "Canada": (
        "1-833-456-4566",
        "Canada Suicide Prevention",
        "https://www.crisisservicescanada.ca",
    ),
    "Australia": ("13 11 14", "Lifeline Australia", "https://www.lifeline.org.au"),
    "Ireland": ("116 123", "Samaritans Ireland", "https://www.samaritans.org"),
    "New Zealand": (
        "0800 543 354",
        "Lifeline New Zealand",
        "https://www.lifeline.org.nz",
    ),
    "South Africa": ("0800 567 567", "SADAG", "https://www.sadag.org"),
}

INTERNATIONAL_FALLBACK = ("000", "Befrienders Worldwide", "https://www.befrienders.org")
CRISIS_TEXT_LINE = (
    "Text HOME to 741741 (Crisis Text Line available in US, UK, Canada, Ireland)"
)

CRISIS_RESOURCES_TEMPLATE = {
    "en": (
        "\n\n**Crisis support — free, confidential, available now:**\n"
        "  * {hotline_name}: {hotline_number}\n"
        "  * Website: {hotline_url}\n"
        "  * International Support: https://www.befrienders.org\n"
        "  * {crisis_text}"
    ),
    "ar": (
        "\n\n**الدعم المتاح في حالات الأزمات — مجاني، سري، ومتوفر الآن:**\n"
        "  * {hotline_name}: {hotline_number}\n"
        "  * الموقع الإلكتروني: {hotline_url}\n"
        "  * الدعم الدولي: https://www.befrienders.org\n"
        "  * {crisis_text}"
    ),
}


def get_hotline(country: str) -> dict:
    country_clean = country.strip().title() if country else "United States"
    match = CRISIS_HOTLINES.get(country_clean, INTERNATIONAL_FALLBACK)
    number, name, url = match
    return {
        "country": country_clean,
        "hotline_number": number,
        "hotline_name": name,
        "hotline_url": url,
        "crisis_text": CRISIS_TEXT_LINE,
    }


def log_crisis_event(db: Session, session_id: str, trigger_text: str) -> CrisisEvent:
    event = CrisisEvent(
        session_id=session_id,
        trigger_text=trigger_text[:1000],
        detected_at=datetime.now(timezone.utc),
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event
