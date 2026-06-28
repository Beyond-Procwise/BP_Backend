"""Negotiation email HTML builders — pure presentation, extracted from the
NegotiationAgent god-class. No business logic, no DB; only stdlib + typing.

Public classes (re-exported by agents.negotiation_agent for backward-compat):
  - NegotiationEmailHTMLShellBuilder
  - NegotiationEmailHTMLBuilder
"""
from __future__ import annotations

import re
from html import escape
from typing import Any, Dict, List, Optional


class NegotiationEmailHTMLShellBuilder:
    """Render negotiation drafts into a modern email-safe HTML shell."""

    BRAND_LABEL = "Beyond Procwise"
    MAX_PREHEADER = 160
    PARAGRAPH_STYLE = (
        "margin:0 0 16px 0;" "font-size:15px;" "line-height:1.6;" "color:#1f2937;"
    )
    LIST_STYLE = "margin:0 0 16px 24px;padding:0;"
    LIST_ITEM_STYLE = (
        "margin:0 0 8px 0;" "padding:0;" "font-size:15px;" "line-height:1.6;" "color:#1f2937;"
    )

    def __init__(self, *, brand_label: Optional[str] = None) -> None:
        self.brand_label = (brand_label or self.BRAND_LABEL).strip() or self.BRAND_LABEL

    @staticmethod
    def _parse_blocks(text: str) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        bullets: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                if bullets:
                    blocks.append({"type": "list", "items": list(bullets)})
                    bullets.clear()
                continue
            bullet_match = re.match(r"^[-*•]\s+(.*)$", stripped)
            if bullet_match:
                bullets.append(bullet_match.group(1).strip())
                continue
            if bullets:
                blocks.append({"type": "list", "items": list(bullets)})
                bullets.clear()
            blocks.append({"type": "paragraph", "text": stripped})
        if bullets:
            blocks.append({"type": "list", "items": list(bullets)})
        return blocks

    @classmethod
    def _render_blocks(cls, blocks: List[Dict[str, Any]]) -> str:
        html_parts: List[str] = []
        for block in blocks:
            block_type = block.get("type")
            if block_type == "list":
                items = block.get("items") or []
                if not isinstance(items, Sequence):
                    continue
                rendered_items = "".join(
                    f'<li style="{cls.LIST_ITEM_STYLE}">{escape(str(item))}</li>'
                    for item in items
                )
                if rendered_items:
                    html_parts.append(f'<ul style="{cls.LIST_STYLE}">{rendered_items}</ul>')
            elif block_type == "paragraph":
                text = block.get("text")
                if not isinstance(text, str):
                    continue
                html_parts.append(f'<p style="{cls.PARAGRAPH_STYLE}">{escape(text)}</p>')
        return "".join(html_parts)

    @staticmethod
    def _build_preheader(text: str) -> str:
        if not text:
            return ""
        collapsed = re.sub(r"\s+", " ", text).strip()
        if not collapsed:
            return ""
        if len(collapsed) <= NegotiationEmailHTMLShellBuilder.MAX_PREHEADER:
            return collapsed
        truncated = collapsed[: NegotiationEmailHTMLShellBuilder.MAX_PREHEADER - 1].rstrip()
        return f"{truncated}…"

    def build(self, *, subject: Optional[str], body_text: str, preheader: Optional[str] = None) -> str:
        if not isinstance(body_text, str):
            return ""
        trimmed_body = body_text.strip()
        if not trimmed_body:
            return ""

        safe_subject = escape((subject or "Negotiation Update").strip() or "Negotiation Update")
        blocks = self._parse_blocks(trimmed_body)
        body_html = self._render_blocks(blocks)
        if not body_html:
            return ""

        preheader_source = preheader if isinstance(preheader, str) else trimmed_body
        preheader_text = escape(self._build_preheader(preheader_source))
        brand_html = escape(self.brand_label)

        return (
            "<!DOCTYPE html>\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            "  <meta charset=\"utf-8\"/>\n"
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\"/>\n"
            f"  <title>{safe_subject}</title>\n"
            "</head>\n"
            '<body style="margin:0;padding:0;background-color:#f5f6fa;">\n'
            f"  <div style=\"display:none;font-size:1px;color:#f5f6fa;line-height:1px;max-height:0;max-width:0;opacity:0;overflow:hidden;\">{preheader_text}</div>\n"
            '  <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="background-color:#f5f6fa;">\n'
            "    <tr>\n"
            '      <td align="center" style="padding:24px 16px;">\n'
            '        <table role="presentation" cellpadding="0" cellspacing="0" width="100%" style="max-width:640px;background-color:#ffffff;border-radius:14px;border:1px solid #e2e8f0;overflow:hidden;">\n'
            "          <tr>\n"
            '            <td style="padding:24px 32px;background-color:#0f172a;color:#f8fafc;font-family:\'Segoe UI\',Arial,sans-serif;font-size:16px;font-weight:600;letter-spacing:0.02em;">\n'
            f"              {brand_html}\n"
            "            </td>\n"
            "          </tr>\n"
            "          <tr>\n"
            '            <td style="padding:24px 32px 8px 32px;font-family:\'Segoe UI\',Arial,sans-serif;font-size:22px;font-weight:600;color:#0f172a;">\n'
            f"              {safe_subject}\n"
            "            </td>\n"
            "          </tr>\n"
            "          <tr>\n"
            '            <td style="padding:0 32px 32px 32px;font-family:\'Segoe UI\',Arial,sans-serif;">\n'
            f"              {body_html}\n"
            "            </td>\n"
            "          </tr>\n"
            "          <tr>\n"
            '            <td style="padding:20px 32px 28px 32px;background-color:#f8fafc;font-family:\'Segoe UI\',Arial,sans-serif;font-size:12px;line-height:1.6;color:#64748b;border-top:1px solid #e2e8f0;">\n'
            "              <p style=\"margin:0 0 6px 0;font-weight:600;color:#0f172a;\">Human review required</p>\n"
            "              <p style=\"margin:0;\">This negotiation draft was prepared by Beyond Procwise’s agentic framework and must be validated by a procurement lead before sending to the supplier.</p>\n"
            "            </td>\n"
            "          </tr>\n"
            "        </table>\n"
            "      </td>\n"
            "    </tr>\n"
            "  </table>\n"
            "</body>\n"
            "</html>"
        )


class NegotiationEmailHTMLBuilder:
    """Build professional HTML emails for negotiation rounds."""

    BASE_STYLES = """
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333333;
            max-width: 650px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .email-container {
            background-color: #ffffff;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 30px;
            margin: 20px auto;
        }
        .header {
            border-bottom: 3px solid #0066cc;
            padding-bottom: 15px;
            margin-bottom: 25px;
        }
        .header h1 {
            color: #0066cc;
            font-size: 24px;
            margin: 0 0 5px 0;
            font-weight: 600;
        }
        .round-badge {
            display: inline-block;
            background-color: #0066cc;
            color: #ffffff;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .greeting {
            font-size: 16px;
            margin-bottom: 20px;
            color: #333333;
        }
        .section {
            margin: 25px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-left: 4px solid #0066cc;
            border-radius: 4px;
        }
        .section-title {
            font-size: 18px;
            font-weight: 600;
            color: #0066cc;
            margin: 0 0 12px 0;
        }
        .pricing-table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            background-color: #ffffff;
            border-radius: 6px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .pricing-table th {
            background-color: #0066cc;
            color: #ffffff;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            font-size: 14px;
        }
        .pricing-table td {
            padding: 12px;
            border-bottom: 1px solid #e9ecef;
            font-size: 14px;
        }
        .pricing-table tr:last-child td {
            border-bottom: none;
        }
        .pricing-table .label {
            font-weight: 600;
            color: #495057;
        }
        .pricing-table .value {
            color: #212529;
        }
        .pricing-table .highlight {
            background-color: #fff3cd;
            font-weight: 600;
            color: #856404;
        }
        .asks-list {
            margin: 15px 0;
            padding: 0;
            list-style: none;
        }
        .asks-list li {
            padding: 10px 15px;
            margin: 8px 0;
            background-color: #ffffff;
            border-left: 3px solid #28a745;
            border-radius: 4px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }
        .asks-list li:before {
            content: "✓";
            color: #28a745;
            font-weight: bold;
            margin-right: 10px;
        }
        .callout {
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 6px;
            font-size: 14px;
        }
        .callout-info {
            background-color: #d1ecf1;
            border-left: 4px solid #0c5460;
            color: #0c5460;
        }
        .callout-warning {
            background-color: #fff3cd;
            border-left: 4px solid #856404;
            color: #856404;
        }
        .callout-success {
            background-color: #d4edda;
            border-left: 4px solid #155724;
            color: #155724;
        }
        .playbook-recommendations {
            margin: 20px 0;
            padding: 0;
        }
        .recommendation-item {
            padding: 15px;
            margin: 10px 0;
            background-color: #e7f3ff;
            border-left: 4px solid #0066cc;
            border-radius: 4px;
        }
        .recommendation-item .lever {
            font-weight: 600;
            color: #0066cc;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }
        .recommendation-item .description {
            color: #495057;
            font-size: 14px;
            line-height: 1.5;
        }
        .footer {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 2px solid #e9ecef;
            font-size: 14px;
            color: #6c757d;
        }
        .signature {
            margin-top: 25px;
            font-size: 15px;
            color: #333333;
        }
        .signature .name {
            font-weight: 600;
            color: #0066cc;
        }
        .button {
            display: inline-block;
            padding: 12px 24px;
            background-color: #0066cc;
            color: #ffffff !important;
            text-decoration: none;
            border-radius: 6px;
            font-weight: 600;
            font-size: 14px;
            text-align: center;
            margin: 15px 0;
        }
        .button:hover {
            background-color: #0052a3;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .metric-card {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .metric-card .label {
            font-size: 12px;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }
        .metric-card .value {
            font-size: 20px;
            font-weight: 600;
            color: #0066cc;
        }
        @media only screen and (max-width: 600px) {
            body {
                padding: 10px;
            }
            .email-container {
                padding: 20px;
            }
            .pricing-table {
                font-size: 12px;
            }
            .metrics-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """

    @staticmethod
    def build_negotiation_email(
        *,
        round_number: int,
        contact_name: Optional[str],
        supplier_name: Optional[str],
        decision: Dict[str, Any],
        positions: Optional[Dict[str, Any]],
        currency: Optional[str],
        playbook_recommendations: Optional[List[Dict[str, Any]]],
        negotiation_message: str,
        sender_name: Optional[str] = None,
        company_name: Optional[str] = "Procwise",
    ) -> str:
        strategy = decision.get("strategy", "counter")
        counter_price = decision.get("counter_price")
        asks = decision.get("asks", [])
        lead_time_request = decision.get("lead_time_request")

        greeting = NegotiationEmailHTMLBuilder._build_greeting(
            contact_name=contact_name,
            supplier_name=supplier_name,
            round_number=round_number,
        )

        header = NegotiationEmailHTMLBuilder._build_header(
            round_number=round_number,
            strategy=strategy,
        )

        pricing_section = ""
        if counter_price is not None and positions:
            pricing_section = NegotiationEmailHTMLBuilder._build_pricing_section(
                counter_price=counter_price,
                positions=positions,
                currency=currency,
            )

        message_section = NegotiationEmailHTMLBuilder._build_message_section(
            negotiation_message=negotiation_message,
            round_number=round_number,
        )

        asks_section = ""
        if asks or lead_time_request:
            asks_section = NegotiationEmailHTMLBuilder._build_asks_section(
                asks=asks,
                lead_time_request=lead_time_request,
            )

        playbook_section = ""
        if playbook_recommendations:
            playbook_section = NegotiationEmailHTMLBuilder._build_playbook_section(
                recommendations=playbook_recommendations[:3],
            )

        footer = NegotiationEmailHTMLBuilder._build_footer(
            sender_name=sender_name,
            company_name=company_name,
            round_number=round_number,
        )

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <title>Negotiation Round {round_number}</title>
    {NegotiationEmailHTMLBuilder.BASE_STYLES}
</head>
<body>
    <div class="email-container">
        {header}
        {greeting}
        {message_section}
        {pricing_section}
        {asks_section}
        {playbook_section}
        {footer}
    </div>
</body>
</html>
        """

        return html

    @staticmethod
    def _build_greeting(
        *,
        contact_name: Optional[str],
        supplier_name: Optional[str],
        round_number: int,
    ) -> str:
        name = contact_name or supplier_name or "there"

        if round_number == 1:
            greeting_text = f"Dear {name},"
        elif round_number == 2:
            greeting_text = f"Dear {name},<br><br>Thank you for your response."
        else:
            greeting_text = (
                f"Dear {name},<br><br>Thank you for continuing our discussion."
            )

        return f'<div class="greeting">{greeting_text}</div>'

    @staticmethod
    def _build_header(*, round_number: int, strategy: str) -> str:
        strategy_titles = {
            "counter": "Negotiation Proposal",
            "accept": "Agreement Confirmation",
            "decline": "Negotiation Status",
            "clarify": "Request for Information",
            "review": "Review Required",
        }

        title = strategy_titles.get(strategy, "Negotiation Update")

        return f"""
        <div class="header">
            <h1>{title}</h1>
            <span class="round-badge">Round {round_number}</span>
        </div>
        """

    @staticmethod
    def _build_pricing_section(
        *,
        counter_price: float,
        positions: Dict[str, Any],
        currency: Optional[str],
    ) -> str:
        def format_price(value: Optional[float]) -> str:
            if value is None:
                return "—"
            symbol = (
                "£"
                if currency == "GBP"
                else "$"
                if currency == "USD"
                else "€"
                if currency == "EUR"
                else ""
            )
            return f"{symbol}{value:,.2f}"

        current_offer = positions.get("supplier_offer")
        target = positions.get("desired")

        rows = []

        if current_offer is not None:
            rows.append(
                f"""
            <tr>
                <td class="label">Your Current Offer</td>
                <td class="value">{format_price(current_offer)}</td>
            </tr>
            """
            )

        rows.append(
            f"""
        <tr class="highlight">
            <td class="label">Our Counter Proposal</td>
            <td class="value">{format_price(counter_price)}</td>
        </tr>
        """
        )

        if target is not None:
            rows.append(
                f"""
            <tr>
                <td class="label">Target Price</td>
                <td class="value">{format_price(target)}</td>
            </tr>
            """
            )

        if current_offer is not None and counter_price is not None:
            gap = current_offer - counter_price
            gap_pct = (gap / current_offer) * 100 if current_offer else 0
            rows.append(
                f"""
            <tr>
                <td class="label">Price Adjustment</td>
                <td class="value">{format_price(gap)} ({gap_pct:.1f}%)</td>
            </tr>
            """
            )

        return f"""
        <div class="section">
            <div class="section-title">💰 Pricing Proposal</div>
            <table class="pricing-table">
                <thead>
                    <tr>
                        <th>Item</th>
                        <th>Amount</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """

    @staticmethod
    def _build_message_section(
        *,
        negotiation_message: str,
        round_number: int,
    ) -> str:
        paragraphs = negotiation_message.split("\n\n")
        html_paragraphs = [
            f"<p>{escape(p.strip())}</p>" for p in paragraphs if p.strip()
        ]

        callout_type = "callout-info" if round_number <= 2 else "callout-warning"

        return f"""
        <div class="section">
            <div class="section-title">📋 Proposal Details</div>
            <div class="callout {callout_type}">
                {''.join(html_paragraphs)}
            </div>
        </div>
        """

    @staticmethod
    def _build_asks_section(
        *,
        asks: List[str],
        lead_time_request: Optional[str],
    ) -> str:
        items: List[str] = []

        if lead_time_request:
            items.append(
                f"<li><strong>Lead Time:</strong> {escape(lead_time_request)}</li>"
            )

        for ask in asks:
            if ask and isinstance(ask, str):
                items.append(f"<li>{escape(ask)}</li>")

        if not items:
            return ""

        return f"""
        <div class="section">
            <div class="section-title">✓ Key Requirements</div>
            <ul class="asks-list">
                {''.join(items)}
            </ul>
        </div>
        """

    @staticmethod
    def _build_playbook_section(
        *,
        recommendations: List[Dict[str, Any]],
    ) -> str:
        if not recommendations:
            return ""

        items: List[str] = []
        for rec in recommendations[:3]:
            if not isinstance(rec, dict):
                continue

            lever = escape(str(rec.get("lever", "")))
            description = escape(str(rec.get("play", "")))

            if lever and description:
                items.append(
                    f"""
                <div class="recommendation-item">
                    <div class="lever">{lever}</div>
                    <div class="description">{description}</div>
                </div>
                """
                )

        if not items:
            return ""

        return f"""
        <div class="section">
            <div class="section-title">💡 Value Creation Opportunities</div>
            <div class="playbook-recommendations">
                {''.join(items)}
            </div>
        </div>
        """

    @staticmethod
    def _build_footer(
        *,
        sender_name: Optional[str],
        company_name: str,
        round_number: int,
    ) -> str:
        sender = sender_name or "The Procurement Team"

        if round_number >= 3:
            next_steps = """
            <div class="callout callout-warning">
                <strong>⏰ Closing Round:</strong> We're working to finalize this agreement. \
                Please respond at your earliest convenience to keep the process moving forward.
            </div>
            """
        else:
            next_steps = """
            <div class="callout callout-info">
                <strong>Next Steps:</strong> We look forward to your response and are happy to \
                discuss any aspects of this proposal in more detail.
            </div>
            """

        return f"""
        {next_steps}

        <div class="signature">
            <p>Best regards,<br>
            <span class="name">{escape(sender)}</span><br>
            {escape(company_name)}</p>
        </div>

        <div class="footer">
            <p style="font-size: 12px; color: #6c757d;">
                This is an automated negotiation communication generated as part of our procurement workflow.
                For questions or concerns, please reply directly to this email.
            </p>
        </div>
        """
