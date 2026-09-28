"""Interface strings for the workbench.

Three things follow from the specification's stated adoption barrier - a
lack of trust in an opaque tool, and a world whose analysts do not all read
English - and all three are implemented here rather than left as a TODO:

* **A missing translation falls back to English**, never to a blank label.
  An unlabelled control is unusable with a screen reader, so a partial
  translation must degrade to *working*, not to *broken*.
* **The strings carry context, not just words.** "Fallbacks" is a noun;
  "why the system chose something else" is what an analyst reads at a
  glance.
* **Right-to-left is a layout concern**, so ``dir`` is a per-language
  attribute here rather than something the CSS has to guess.

Adding a language is one dict entry. Nothing else changes.
"""

from __future__ import annotations

from typing import Any

__all__ = ["STRINGS", "LANGUAGES", "text_direction"]


#: Language tag -> display direction. Anything unlisted is left-to-right.
DIRECTION: dict[str, str] = {
    "ar": "rtl", "he": "rtl", "fa": "rtl", "ur": "rtl",
}

EN: dict[str, str] = {
    "app.title": "AAR Analyst Workbench",
    "tab.project": "Project", "tab.connections": "Connections",
    "tab.runs": "Runs", "tab.resources": "Resources", "tab.help": "Help",
    "panel.sources": "Sources", "panel.workspace": "SQL / Python workspace",
    "panel.inspector": "Plan inspector", "panel.preview": "Data preview",
    "panel.quality": "Quality", "panel.schema": "Schema",
    "panel.lineage": "Lineage", "panel.log": "Execution log",
    "panel.explain": "Why",
    "label.engine": "Engine", "label.available": "available",
    "label.unavailable": "not installed", "label.pipeline": "Pipeline file",
    "label.role": "Your role",
    "action.explain": "Explain plan", "action.run": "Run",
    "action.refresh": "Refresh",
    "label.language": "Language", "label.theme": "Theme",
    "label.density": "Density",
    "theme.dark": "Dark", "theme.light": "Light", "theme.system": "System",
    "density.compact": "Compact", "density.comfortable": "Comfortable",
    "density.spacious": "Spacious",
    "empty.engines": "No engines probed yet.",
    "empty.plan": "Load a pipeline and choose Explain to see why each step "
                  "ran where it did.",
    "empty.result": "Run a pipeline to see its result here.",
    "status.running": "Working...", "status.ready": "Ready",
    "status.error": "Something went wrong",
    "legend.chosen": "chosen", "legend.considered": "also considered",
    "legend.fallback": "fell back because",
    "tag.confidential": "confidential", "tag.public": "public",
    "help.shortcuts": "Keyboard: Ctrl+E explain, Ctrl+R run, "
                      "Ctrl+1..5 switch bottom panel, ? opens help.",
}


def _merge(**kwargs: Any) -> dict[str, str]:
    """A language table, inheriting anything it does not translate."""
    merged = dict(EN)
    merged.update({k: v for k, v in kwargs.items() if v})
    return merged


STRINGS: dict[str, dict[str, str]] = {"en": EN}
LANGUAGES: dict[str, str] = {"en": "English"}

STRINGS.update({
    "es": _merge(**{
        "app.title": "AAR Banco de trabajo del analista",
        "tab.project": "Proyecto", "tab.connections": "Conexiones",
        "tab.runs": "Ejecuciones", "tab.resources": "Recursos",
        "tab.help": "Ayuda", "panel.sources": "Origenes",
        "panel.workspace": "Espacio SQL / Python",
        "panel.inspector": "Inspector de plan",
        "panel.preview": "Vista de datos", "panel.quality": "Calidad",
        "panel.schema": "Esquema", "panel.lineage": "Linaje",
        "panel.log": "Registro de ejecucion", "panel.explain": "Por que",
        "label.engine": "Motor", "label.available": "disponible",
        "label.unavailable": "no instalado",
        "label.pipeline": "Archivo de canalizacion", "label.role": "Su rol",
        "action.explain": "Explicar plan", "action.run": "Ejecutar",
        "action.refresh": "Actualizar", "label.language": "Idioma",
        "label.theme": "Tema", "label.density": "Densidad",
        "theme.dark": "Oscuro", "theme.light": "Claro",
        "theme.system": "Sistema", "density.compact": "Compacta",
        "density.comfortable": "Comoda", "density.spacious": "Amplia",
        "empty.engines": "Sin motores comprobados.",
        "empty.plan": "Cargue una canalizacion y elija Explicar.",
        "empty.result": "Ejecute una canalizacion para ver su resultado.",
        "status.running": "Trabajando...", "status.ready": "Listo",
        "status.error": "Algo salio mal",
        "legend.chosen": "elegido",
        "legend.considered": "tambien considerado",
        "legend.fallback": "recurrio porque",
    }),
    "fr": _merge(**{
        "app.title": "AAR Atelier de l'analyste",
        "tab.project": "Projet", "tab.connections": "Connexions",
        "tab.runs": "Executions", "tab.resources": "Ressources",
        "tab.help": "Aide", "panel.sources": "Sources",
        "panel.inspector": "Inspecteur de plan", "panel.preview": "Apercu",
        "panel.lineage": "Lineage", "panel.explain": "Pourquoi",
        "label.engine": "Moteur", "label.available": "disponible",
        "label.unavailable": "non installe", "label.pipeline": "Fichier pipeline",
        "label.role": "Votre role", "action.explain": "Expliquer le plan",
        "action.run": "Executer", "action.refresh": "Actualiser",
        "label.language": "Langue", "label.density": "Densite",
        "theme.dark": "Sombre", "theme.light": "Clair",
        "status.running": "En cours...", "status.ready": "Pret",
        "legend.chosen": "retenu", "legend.considered": "aussi considere",
    }),
    "hi": _merge(**{
        "app.title": "AAR विश्लेषक कार्यस्थल",
        "tab.project": "परियोजना", "tab.connections": "कनेक्शन",
        "tab.runs": "चलान", "tab.resources": "संसाधन", "tab.help": "सहायता",
        "panel.sources": "स्रोत", "panel.workspace": "SQL / Python कार्यक्षेत्र",
        "panel.inspector": "योजना निरीक्षक", "panel.preview": "डेटा पूर्वावलोकन",
        "panel.quality": "गुणवत्ता", "panel.schema": "स्कीमा",
        "panel.lineage": "वंश", "panel.log": "निष्पादन लॉग",
        "panel.explain": "क्यों", "label.engine": "इंजन",
        "label.available": "उपलब्ध", "label.unavailable": "स्थापित नहीं",
        "label.pipeline": "पाइपलाइन फ़ाइल", "label.role": "आपकी भूमिका",
        "action.explain": "योजना समझाएँ", "action.run": "चलाएँ",
        "action.refresh": "ताज़ा करें", "label.language": "भाषा",
        "label.density": "घनत्व", "theme.dark": "अँधेरा", "theme.light": "हल्का",
        "density.compact": "सघन", "density.comfortable": "आरामदायक",
        "density.spacious": "विस्तृत",
        "empty.engines": "कोई इंजन जाँचा नहीं गया।",
        "empty.plan": "पाइपलाइन लोड करें और समझाएँ चुनें।",
        "empty.result": "परिणाम देखने के लिए पाइपलाइन चलाएँ।",
        "status.running": "काम चल रहा है...", "status.ready": "तैयार",
        "status.error": "कुछ गलत हो गया",
        "legend.chosen": "चुना गया", "legend.fallback": "वापस क्यों गया",
    }),
    "ar": _merge(**{
        "app.title": "AAR مساحة عمل المحلل",
        "tab.project": "المشروع", "tab.connections": "الاتصالات",
        "tab.runs": "التشغيلات", "tab.resources": "الموارد",
        "tab.help": "مساعدة", "panel.sources": "المصادر",
        "panel.workspace": "مساحة SQL / Python",
        "panel.inspector": "فاحص الخطة", "panel.preview": "معاينة البيانات",
        "panel.quality": "الجودة", "panel.schema": "المخطط",
        "panel.lineage": "النسب", "panel.log": "سجل التنفيذ",
        "panel.explain": "لماذا", "label.engine": "المحرك",
        "label.available": "متاح", "label.unavailable": "غير مثبت",
        "label.pipeline": "ملف خط الأنابيب", "label.role": "دورك",
        "action.explain": "اشرح الخطة", "action.run": "تشغيل",
        "action.refresh": "تحديث", "label.language": "اللغة",
        "label.density": "الكثافة", "theme.dark": "داكن",
        "theme.light": "فاتح", "density.compact": "مضغوط",
        "density.comfortable": "مريح", "density.spacious": "واسع",
        "empty.engines": "لم يتم فحص أي محرك.",
        "empty.plan": "قم بتحميل خط أنابيب واختر اشرح.",
        "empty.result": "قم بتشغيل خط أنابيب لعرض النتيجة.",
        "status.running": "جارٍ العمل...", "status.ready": "جاهز",
        "status.error": "حدث خطأ ما", "legend.chosen": "مختار",
        "legend.fallback": "الرجوع لأن",
    }),
})

LANGUAGES.update({
    "es": "Espanol", "fr": "Francais", "hi": "Hindi", "ar": "Arabic",
})


def text_direction(language: str) -> str:
    """`rtl` or `ltr` for a language tag. Unknown languages are `ltr`."""
    return DIRECTION.get((language or "en").split("-")[0].lower(), "ltr")

