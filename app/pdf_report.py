import json
import os
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
from fpdf import FPDF
from groq import Groq

# Monochrome palette only
BLACK = (0, 0, 0)
DARK = (35, 35, 35)
GRAY = (90, 90, 90)
LIGHT_GRAY = (210, 210, 210)
WHITE = (255, 255, 255)
BORDER = (170, 170, 170)

MARGIN = 14
PAD = 8
SECTION_GAP = 8
SECTION_TITLE_H = 9
FOOTER_RESERVE = 12
RECOMMENDATION_RESERVE = 28

H = {
    "header": 28,
    "metrics": 32,
    "charts": 78,
    "risk": 38,
    "columns": 72,
    "recommendation": 24,
}

_client_groq: Groq | None = None


def _get_groq_api_key() -> str | None:
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        return api_key
    try:
        import streamlit as st

        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


def _get_groq_client() -> Groq | None:
    global _client_groq
    api_key = _get_groq_api_key()
    if not api_key:
        return None
    if _client_groq is None:
        _client_groq = Groq(api_key=api_key)
    return _client_groq


def _risk_message(fraud_count: int, anomaly_count: int) -> str:
    if fraud_count > 0:
        return (
            f"{fraud_count:,} confirmed fraud transaction(s) detected - "
            "immediate review required."
        )
    if anomaly_count > 0:
        return (
            f"{anomaly_count:,} suspicious transaction(s) - "
            "none confirmed as fraud at the current threshold."
        )
    return "No anomalies detected. All transactions appear within normal patterns."


def _fallback_recommendation(risk: str, fraud_count: int, anomaly_count: int) -> str:
    if risk == "CRITICAL" or fraud_count > 0:
        return (
            "Prioritize immediate review of confirmed fraud transactions. "
            "Strengthen fraud detection rules and monitor high-risk patterns."
        )
    if risk == "HIGH" or anomaly_count > 0:
        return (
            "Review suspicious transactions closely. Consider lowering the "
            "fraud threshold and enabling enhanced monitoring."
        )
    return (
        "Continue routine monitoring. Current transaction patterns appear "
        "stable with no immediate action required."
    )


def _build_analysis_context(
    *,
    file_name: str,
    total: int,
    clean_count: int,
    normal_pct: float,
    anomaly_count: int,
    anomaly_pct: float,
    fraud_count: int,
    fraud_pct: float,
    safe_susp: int,
    avg_fraud_score: float,
    risk: str,
    insights: list[tuple[str, str]],
) -> dict:
    context = {
        "system": "KAVACAPay Fraud Detection System",
        "dataset": file_name,
        "overall_risk_level": risk,
        "total_transactions": total,
        "normal_transactions": {
            "count": clean_count,
            "percentage": round(normal_pct, 2),
        },
        "anomalies_detected": {
            "count": anomaly_count,
            "percentage": round(anomaly_pct, 2),
        },
        "confirmed_fraud": {
            "count": fraud_count,
            "percentage": round(fraud_pct, 4),
        },
        "suspicious_not_confirmed": safe_susp,
        "average_fraud_score": round(avg_fraud_score, 4),
        "risk_assessment": _risk_message(fraud_count, anomaly_count),
        "key_insights": [
            {"title": title, "description": desc}
            for title, desc in insights
        ],
    }
    if fraud_count and anomaly_count:
        context["fraud_precision_among_anomalies"] = round(
            fraud_count / anomaly_count * 100, 1
        )
    return context


def _recommendation(data) -> str:
    if isinstance(data, str):
        analysis = data
    elif isinstance(data, pd.DataFrame):
        analysis = data.to_string(index=False)
    elif isinstance(data, dict):
        analysis = json.dumps(data, indent=2)
    elif isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], tuple):
            analysis = "\n".join(f"{k}: {v}" for k, v in data)
        else:
            analysis = "\n".join(map(str, data))
    else:
        analysis = str(data)

    client = _get_groq_client()
    if client is None:
        if isinstance(data, dict):
            risk = data.get("overall_risk_level", "LOW")
            fraud_count = data.get("confirmed_fraud", {}).get("count", 0)
            anomaly_count = data.get("anomalies_detected", {}).get("count", 0)
            return _fallback_recommendation(risk, fraud_count, anomaly_count)
        return (
            "Continue routine monitoring. Configure GROQ_API_KEY to enable "
            "AI-generated recommendations for this report."
        )

    prompt = f"""
Role: You are an AI assistant that generates the conclusion section for fraud detection reports.

Task:
Generate a concise conclusion based strictly on the provided analysis from a credit card fraud detection system (Isolation Forest + Random Forest pipeline).

Rules:
1. Do not fabricate information.
2. Do not repeat every metric.
3. Mention the overall risk level naturally.
4. Explain what the results indicate about transaction integrity.
5. Mention possible business impact (financial loss, customer trust, compliance).
6. Recommend practical next steps for the fraud analyst.
7. Keep the response between 70 and 100 words.
8. Write in a formal, professional tone.
9. Output only the conclusion text.

Analysis Data:

{analysis}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=350,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        if isinstance(data, dict):
            risk = data.get("overall_risk_level", "LOW")
            fraud_count = data.get("confirmed_fraud", {}).get("count", 0)
            anomaly_count = data.get("anomalies_detected", {}).get("count", 0)
            return _fallback_recommendation(risk, fraud_count, anomaly_count)
        return (
            "Unable to generate an AI recommendation at this time. "
            "Please review the risk assessment and key insights above."
        )


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _is_bimodal(fraud_probs) -> bool:
    probs = np.asarray(fraud_probs)
    if len(probs) == 0:
        return False
    return (probs < 0.2).mean() > 0.25 and (probs > 0.8).mean() > 0.25


def _build_insights(
    *,
    fraud_pct: float,
    anomaly_pct: float,
    safe_susp: int,
    fraud_count: int,
    bimodal: bool,
) -> list[tuple[str, str]]:
    insights: list[tuple[str, str]] = []

    if fraud_pct > 0.2:
        insights.append((
            "High Fraud Rate Detected",
            f"Fraud rate at {fraud_pct:.4f}% requires immediate attention.",
        ))
    elif fraud_count > 0:
        insights.append((
            "Fraud Cases Identified",
            f"{fraud_count:,} transactions confirmed as fraudulent.",
        ))

    if anomaly_pct > 3:
        insights.append((
            "Significant Anomalies",
            f"{anomaly_pct:.2f}% of transactions show unusual patterns.",
        ))
    elif anomaly_pct > 0:
        insights.append((
            "Anomalies Present",
            f"{anomaly_pct:.2f}% of transactions flagged as statistical outliers.",
        ))

    if bimodal:
        insights.append((
            "Bi-Modal Distribution",
            "Fraud scores cluster near both extremes (0 and 1).",
        ))
    else:
        insights.append((
            "Score Distribution",
            "Fraud probability scores are spread across the transaction set.",
        ))

    if safe_susp == 0 and anomaly_pct > 0:
        insights.append((
            "Zero Suspicious Transactions",
            "All anomalies were confirmed as fraud or classified as clean.",
        ))
    elif safe_susp > 0:
        insights.append((
            "Suspicious Activity",
            f"{safe_susp:,} transactions flagged suspicious but not confirmed.",
        ))
    else:
        insights.append((
            "Clean Dataset",
            "No anomalies or fraud detected in the current analysis.",
        ))

    return insights[:4]


def _chart_to_png(fig) -> BytesIO:
    buf = BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=160,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    buf.seek(0)
    return buf


class FraudReportPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="L", unit="mm", format="A4")
        self.set_margins(MARGIN, MARGIN, MARGIN)
        self.set_auto_page_break(auto=False)
        self._cursor_y = MARGIN
        self._total_pages = 2

    def add_page(self, *args, **kwargs):
        super().add_page(*args, **kwargs)
        self._cursor_y = MARGIN

    @property
    def content_width(self) -> float:
        return self.w - self.l_margin - self.r_margin

    @property
    def page_bottom(self) -> float:
        return self.h - MARGIN

    @property
    def content_bottom(self) -> float:
        return self.h - MARGIN - FOOTER_RESERVE

    def footer(self):
        self.set_y(-10)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 5, f"Page {self.page_no()} of {self._total_pages}", align="C")

    def _advance(self, height: float):
        self._cursor_y += height

    def _section_title(self, title: str):
        y = self._cursor_y
        self.set_xy(self.l_margin, y)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*BLACK)
        self.cell(0, 6, title, ln=True)
        line_y = y + 6.5
        self.set_draw_color(*BLACK)
        self.set_line_width(0.3)
        self.line(self.l_margin, line_y, self.l_margin + 40, line_y)
        self.set_line_width(0.2)
        self._advance(SECTION_TITLE_H)

    def _box(self, x, y, w, h):
        self.set_fill_color(*WHITE)
        self.set_draw_color(*BORDER)
        self.rect(x, y, w, h, style="FD")

    def draw_header(self, file_name: str, user_name: str, generated: str):
        y0 = self._cursor_y
        h = H["header"]
        self._box(self.l_margin, y0, self.content_width, h)

        tx = self.l_margin + PAD
        self.set_xy(tx, y0 + PAD)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*BLACK)
        self.cell(0, 8, "Fraud Detection System", ln=True)

        self.set_x(tx)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*DARK)
        self.cell(0, 6, f"Summary Report - {_truncate(file_name, 50)}", ln=True)

        self.set_x(tx)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*GRAY)
        self.cell(
            0,
            5,
            f"Generated: {generated}    |    Analyst: {user_name}",
            ln=True,
        )

        self._advance(h + SECTION_GAP)

    def draw_metric_row(
        self,
        total: int,
        clean_count: int,
        normal_pct: float,
        anomaly_count: int,
        anomaly_pct: float,
        fraud_count: int,
        fraud_pct: float,
        safe_susp: int,
    ):
        self._section_title("Key Metrics")

        cards = [
            ("Total Transactions", f"{total:,}", "Analyzed"),
            ("Normal Transactions", f"{clean_count:,}", f"{normal_pct:.2f}%"),
            ("Anomalies Detected", f"{anomaly_count:,}", f"{anomaly_pct:.2f}%"),
            ("Confirmed Fraud", f"{fraud_count:,}", f"{fraud_pct:.4f}%"),
            (
                "Suspicious",
                f"{safe_susp:,}",
                "Not Confirmed" if safe_susp else "None Found",
            ),
        ]

        card_gap = 5
        card_w = (self.content_width - card_gap * 4) / 5
        card_h = H["metrics"]
        y0 = self._cursor_y

        for i, (label, value, sub) in enumerate(cards):
            x = self.l_margin + i * (card_w + card_gap)
            self._box(x, y0, card_w, card_h)

            self.set_xy(x + PAD, y0 + PAD)
            self.set_font("Helvetica", "", 7)
            self.set_text_color(*GRAY)
            self.cell(card_w - PAD * 2, 4, label.upper())

            self.set_xy(x + PAD, y0 + PAD + 8)
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(*BLACK)
            self.cell(card_w - PAD * 2, 7, value)

            self.set_xy(x + PAD, y0 + PAD + 17)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*DARK)
            self.cell(card_w - PAD * 2, 5, sub)

        self._advance(card_h + SECTION_GAP)

    def draw_charts(self, fig):
        self._section_title("Visual Analytics")
        panel_h = H["charts"]
        x0, y0 = self.l_margin, self._cursor_y
        self._box(x0, y0, self.content_width, panel_h)

        img_buf = _chart_to_png(fig)
        img_h = panel_h - PAD * 2
        self.image(
            img_buf,
            x=x0 + PAD,
            y=y0 + PAD,
            w=self.content_width - PAD * 2,
            h=img_h,
        )
        self._advance(panel_h + SECTION_GAP)

    def draw_risk_banner(
        self,
        risk: str,
        anomaly_pct: float,
        fraud_pct: float,
        avg_fraud_score: float,
        fraud_count: int,
        anomaly_count: int,
    ):
        self._section_title("Risk Assessment")
        h = H["risk"]
        x0, y0 = self.l_margin, self._cursor_y
        self._box(x0, y0, self.content_width, h)

        self.set_draw_color(*BLACK)
        self.set_line_width(0.5)
        self.line(x0 + PAD, y0 + PAD + 14, x0 + self.content_width - PAD, y0 + PAD + 14)
        self.set_line_width(0.2)

        self.set_xy(x0 + PAD, y0 + PAD)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*GRAY)
        self.cell(50, 5, "Overall Risk Level")

        self.set_xy(x0 + PAD, y0 + PAD + 6)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(*BLACK)
        self.cell(60, 8, risk)

        stats = [
            ("Anomaly Rate", f"{anomaly_pct:.2f}%"),
            ("Fraud Rate", f"{fraud_pct:.4f}%"),
            ("Avg. Fraud Score", f"{avg_fraud_score:.1%}"),
        ]
        stat_w = (self.content_width - PAD * 2 - 10) / 3
        for i, (label, value) in enumerate(stats):
            sx = x0 + PAD + i * (stat_w + 5)
            sy = y0 + PAD
            self._box(sx, sy, stat_w, 12)
            self.set_xy(sx + 3, sy + 2)
            self.set_font("Helvetica", "", 7)
            self.set_text_color(*GRAY)
            self.cell(stat_w - 6, 4, label)
            self.set_xy(sx + 3, sy + 6)
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(*BLACK)
            self.cell(stat_w - 6, 5, value)

        self.set_xy(x0 + PAD, y0 + PAD + 18)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*DARK)
        self.multi_cell(
            self.content_width - PAD * 2,
            5,
            _risk_message(fraud_count, anomaly_count),
        )

        self._advance(h + SECTION_GAP)

    def _wrapped_line_count(
        self, text: str, width_mm: float, font_style: str = "", font_size: int = 7
    ) -> int:
        return len(
            self._wrap_text_lines(text, width_mm, font_style=font_style, font_size=font_size)
        )

    def _wrap_text_lines(
        self,
        text: str,
        width_mm: float,
        font_style: str = "",
        font_size: int = 7,
    ) -> list[str]:
        self.set_font("Helvetica", font_style, font_size)
        if width_mm <= 0 or not text or not text.strip():
            return [""]
        words = text.split()
        lines: list[str] = []
        current = words[0]
        space_w = self.get_string_width(" ")
        for word in words[1:]:
            candidate = f"{current} {word}"
            if self.get_string_width(candidate) <= width_mm:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def _insight_card_height(self, desc: str, text_w: float) -> float:
        title_h = 4
        line_h = 3.2
        pad = 3
        desc_lines = self._wrapped_line_count(desc, text_w, "", 7)
        return pad + title_h + desc_lines * line_h + pad

    def _insights_body_height(
        self,
        insights: list[tuple[str, str]],
        text_w: float,
        insight_gap: float,
    ) -> float:
        if not insights:
            return 0.0
        return (
            sum(self._insight_card_height(desc, text_w) for _, desc in insights)
            + insight_gap * (len(insights) - 1)
        )

    def _fit_insights_to_height(
        self,
        insights: list[tuple[str, str]],
        text_w: float,
        max_body_h: float,
        insight_gap: float,
    ) -> list[tuple[str, str]]:
        if not insights or max_body_h <= 0:
            return []

        fitted: list[tuple[str, str]] = []
        for title, desc in insights:
            trial = fitted + [(title, desc)]
            if self._insights_body_height(trial, text_w, insight_gap) <= max_body_h:
                fitted = trial
                continue

            used = self._insights_body_height(fitted, text_w, insight_gap)
            gap = insight_gap if fitted else 0
            remaining = max_body_h - used - gap
            if remaining < 10:
                break

            max_lines = max(int((remaining - 6) / 3.2), 1)
            short_desc = desc
            added = False
            for lines in range(max_lines, 0, -1):
                short_desc = self._truncate_words_to_lines(desc, text_w, lines, 7)
                trial = fitted + [(title, short_desc)]
                if self._insights_body_height(trial, text_w, insight_gap) <= max_body_h:
                    fitted = trial
                    added = True
                    break
            if not added:
                break
        return fitted

    def _truncate_words_to_lines(
        self, text: str, width_mm: float, max_lines: int, font_size: int
    ) -> str:
        words = text.split()
        if not words:
            return ""
        best = words[0]
        for end in range(1, len(words) + 1):
            candidate = " ".join(words[:end])
            if self._wrapped_line_count(candidate, width_mm, "", font_size) <= max_lines:
                best = candidate
            else:
                break
        if best != text.strip():
            trimmed = best.rstrip(".,; ")
            if trimmed:
                best = trimmed + "..."
        return best

    def draw_summary_and_insights(
        self,
        file_name: str,
        total: int,
        clean_count: int,
        normal_pct: float,
        anomaly_count: int,
        anomaly_pct: float,
        fraud_count: int,
        fraud_pct: float,
        safe_susp: int,
        avg_fraud_score: float,
        insights: list[tuple[str, str]],
    ):
        col_gap = 8
        col_w = (self.content_width - col_gap) / 2
        y0 = self._cursor_y
        box_w = col_w - PAD * 2
        text_w = box_w - 8
        insight_gap = 4

        summary_rows = [
            ("Dataset", _truncate(file_name, 32)),
            ("Total Transactions", f"{total:,}"),
            ("Normal Transactions", f"{clean_count:,} ({normal_pct:.2f}%)"),
            ("Anomalies Detected", f"{anomaly_count:,} ({anomaly_pct:.2f}%)"),
            ("Confirmed Fraud", f"{fraud_count:,} ({fraud_pct:.4f}%)"),
            ("Suspicious (Not Confirmed)", f"{safe_susp:,}"),
            ("Average Fraud Score", f"{avg_fraud_score:.1%}"),
        ]
        if fraud_count and anomaly_count:
            precision = fraud_count / anomaly_count * 100
            summary_rows.append(("Fraud Precision (Anomalies)", f"{precision:.1f}%"))

        insights_body_h = self._insights_body_height(insights, text_w, insight_gap)
        insights_panel_h = PAD + 8 + insights_body_h + PAD
        summary_panel_h = PAD + 8 + len(summary_rows) * 5.5 + PAD
        panel_h = max(H["columns"], insights_panel_h, summary_panel_h)
        max_panel_h = self.content_bottom - y0 - SECTION_GAP - RECOMMENDATION_RESERVE
        panel_h = min(panel_h, max_panel_h)

        insights_body_max = max(panel_h - PAD * 2 - 8, 0)
        fitted_insights = self._fit_insights_to_height(
            insights, text_w, insights_body_max, insight_gap
        )
        if not fitted_insights and insights:
            fitted_insights = [insights[0]]

        sx = self.l_margin
        self._box(sx, y0, col_w, panel_h)
        self.set_xy(sx + PAD, y0 + PAD)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*BLACK)
        self.cell(col_w - PAD * 2, 6, "Summary", ln=True)

        row_h = (panel_h - PAD * 2 - 8) / max(len(summary_rows), 1)
        for i, (label, value) in enumerate(summary_rows):
            ry = y0 + PAD + 8 + i * row_h
            self.set_xy(sx + PAD, ry)
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*GRAY)
            self.cell(62, row_h, label)
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(*BLACK)
            self.cell(col_w - PAD * 2 - 64, row_h, value)

        ix = self.l_margin + col_w + col_gap
        self._box(ix, y0, col_w, panel_h)
        self.set_xy(ix + PAD, y0 + PAD)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*BLACK)
        self.cell(col_w - PAD * 2, 6, "Key Insights", ln=True)

        iy = y0 + PAD + 8
        for title, desc in fitted_insights:
            card_h = self._insight_card_height(desc, text_w)
            box_x = ix + PAD
            self._box(box_x, iy, box_w, card_h)

            self.set_xy(box_x + 3, iy + 3)
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(*BLACK)
            self.cell(text_w, 4, title, ln=True)

            self.set_x(box_x + 3)
            self.set_font("Helvetica", "", 7)
            self.set_text_color(*GRAY)
            self.multi_cell(text_w, 3.2, desc)

            iy += card_h + insight_gap

        self._advance(panel_h + SECTION_GAP)

    def draw_recommendation(self, text: str):
        y0 = self._cursor_y
        rec_pad = 5
        text_w = self.content_width - rec_pad * 2
        title_h = 4.5
        body_font = 10
        body_line = 3.2
        title_gap = 1.5

        h = self.content_bottom - y0
        body_top = y0 + rec_pad + title_h + title_gap
        body_max_h = h - rec_pad - title_h - title_gap - rec_pad
        max_lines = max(int(body_max_h / body_line), 2)

        lines = self._wrap_text_lines(text, text_w, "", body_font)
        if len(lines) > max_lines:
            trimmed = " ".join(
                word
                for line in lines[:max_lines]
                for word in line.split()
            )
            body_text = self._truncate_words_to_lines(
                trimmed, text_w, max_lines, body_font
            )
        else:
            body_text = text.strip()

        self._box(self.l_margin, y0, self.content_width, h)

        self.set_xy(self.l_margin + rec_pad, y0 + rec_pad)
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(*BLACK)
        self.cell(0, title_h, "Recommendation", ln=True)

        self.set_xy(self.l_margin + rec_pad, body_top)
        self.set_font("Helvetica", "", body_font)
        self.set_text_color(*DARK)
        self.multi_cell(text_w, body_line, body_text)
        self._advance(h)


def generate_summary_pdf(
    *,
    file_name: str,
    user_name: str,
    total: int,
    anomaly_count: int,
    fraud_count: int,
    anomaly_pct: float,
    fraud_pct: float,
    avg_fraud_score: float,
    risk: str,
    fig,
    fraud_probabilities=None,
) -> bytes:
    clean_count = total - anomaly_count
    safe_susp = max(anomaly_count - fraud_count, 0)
    normal_pct = 100 - anomaly_pct
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    bimodal = _is_bimodal(fraud_probabilities) if fraud_probabilities is not None else False
    insights = _build_insights(
        fraud_pct=fraud_pct,
        anomaly_pct=anomaly_pct,
        safe_susp=safe_susp,
        fraud_count=fraud_count,
        bimodal=bimodal,
    )

    pdf = FraudReportPDF()

    # ── Page 1: overview + charts ───────────────────────────────────────
    pdf.add_page()
    pdf.draw_header(file_name, user_name, generated)
    pdf.draw_metric_row(
        total, clean_count, normal_pct,
        anomaly_count, anomaly_pct,
        fraud_count, fraud_pct, safe_susp,
    )
    pdf.draw_charts(fig)
    assert pdf._cursor_y <= pdf.page_bottom, (
        f"Page 1 overflow: {pdf._cursor_y:.1f}mm > {pdf.page_bottom:.1f}mm"
    )

    # ── Page 2: risk + summary + recommendation ─────────────────────────
    pdf.add_page()
    pdf.draw_risk_banner(
        risk, anomaly_pct, fraud_pct, avg_fraud_score, fraud_count, anomaly_count,
    )
    pdf.draw_summary_and_insights(
        file_name, total, clean_count, normal_pct,
        anomaly_count, anomaly_pct, fraud_count, fraud_pct,
        safe_susp, avg_fraud_score, insights,
    )
    analysis_context = _build_analysis_context(
        file_name=file_name,
        total=total,
        clean_count=clean_count,
        normal_pct=normal_pct,
        anomaly_count=anomaly_count,
        anomaly_pct=anomaly_pct,
        fraud_count=fraud_count,
        fraud_pct=fraud_pct,
        safe_susp=safe_susp,
        avg_fraud_score=avg_fraud_score,
        risk=risk,
        insights=insights,
    )
    pdf.draw_recommendation(_recommendation(analysis_context))
    assert pdf._cursor_y <= pdf.page_bottom, (
        f"Page 2 overflow: {pdf._cursor_y:.1f}mm > {pdf.page_bottom:.1f}mm"
    )

    return bytes(pdf.output())
