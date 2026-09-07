"""Analytic answers: what is true, decided here; how to say it, decided later.

An analytic question ("top 10 suppliers by spend") has an exact answer. Every
number in it, the period it covers, the currency it is stated in, the share one
supplier holds — all of that is arithmetic over rows this system already holds,
and none of it is a language model's to decide. The packages here own that
half: the typed answer (``models``), the currency it is reported in
(``currency``), and the one place a figure is turned into text
(``formatting``).
"""
