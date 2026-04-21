from src.services.structural_extractor.derivation import rule


@rule("subtotal_from_total_tax", "invoice_amount", ["invoice_total_incl_tax", "tax_amount"])
def _sub_inv(inputs):
    try:
        return round(float(inputs["invoice_total_incl_tax"]) - float(inputs["tax_amount"]), 2)
    except Exception:
        return None


@rule("subtotal_from_total_tax_po", "total_amount", ["total_amount_incl_tax", "tax_amount"])
def _sub_po(inputs):
    try:
        return round(float(inputs["total_amount_incl_tax"]) - float(inputs["tax_amount"]), 2)
    except Exception:
        return None


@rule("tax_amount_from_pct_inv", "tax_amount", ["invoice_amount", "tax_percent"])
def _tax_inv(inputs):
    try:
        return round(float(inputs["invoice_amount"]) * float(inputs["tax_percent"]) / 100, 2)
    except Exception:
        return None


@rule("tax_amount_from_pct_po", "tax_amount", ["total_amount", "tax_percent"])
def _tax_po(inputs):
    try:
        return round(float(inputs["total_amount"]) * float(inputs["tax_percent"]) / 100, 2)
    except Exception:
        return None


@rule("total_from_subtotal_tax_inv", "invoice_total_incl_tax", ["invoice_amount", "tax_amount"])
def _tot_inv(inputs):
    try:
        return round(float(inputs["invoice_amount"]) + float(inputs["tax_amount"]), 2)
    except Exception:
        return None


@rule("total_from_subtotal_tax_po", "total_amount_incl_tax", ["total_amount", "tax_amount"])
def _tot_po(inputs):
    try:
        return round(float(inputs["total_amount"]) + float(inputs["tax_amount"]), 2)
    except Exception:
        return None


@rule("tax_pct_from_amounts_inv", "tax_percent", ["invoice_amount", "tax_amount"])
def _pct_inv(inputs):
    try:
        sub = float(inputs["invoice_amount"])
        if sub == 0:
            return None
        return round(float(inputs["tax_amount"]) / sub * 100, 2)
    except Exception:
        return None


@rule("tax_pct_from_amounts_po", "tax_percent", ["total_amount", "tax_amount"])
def _pct_po(inputs):
    try:
        sub = float(inputs["total_amount"])
        if sub == 0:
            return None
        return round(float(inputs["tax_amount"]) / sub * 100, 2)
    except Exception:
        return None
