"""Email style-learning subsystem.

Learns how a specific person writes email and drafts new mail in that voice.

The design deliberately separates two artifacts:

* a **style profile** — a versioned JSON description of writing HABITS (structure,
  register, lexicon, behaviour) that contains no correspondence, and
* **exemplars** — two or three example emails that illustrate what compliance with
  the profile looks like.

Where the two disagree at generation time the profile governs; the examples
illustrate the specification, they do not override it.

This package holds no send capability. No module here, and no adapter it defines,
may expose a method that transmits mail — the platform's existing SES dispatch path
(``services.email_service``) is a separate, deliberately-retained subsystem and must
never be reached from here. See ``docs/style-engine/existing-email-inventory.md``.
"""
