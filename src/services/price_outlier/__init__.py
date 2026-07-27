"""Flag extreme prices for human review. See rule.py for the decision,
detector.py for the database pass."""
from services.price_outlier.rule import OutlierSettings, Verdict, assess

__all__ = ["OutlierSettings", "Verdict", "assess"]
