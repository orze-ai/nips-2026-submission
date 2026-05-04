"""Orze-pro skills — SOP registry and validation.

SOPs (Standard Operating Procedures) are first-class behavioral units that
compose into role prompts. This subpackage is the SOP-specific half of the
skills infrastructure:

- registry.py: SkillMetadata + discover_skills + validate_wiring

Generic skill composition machinery (loader, trigger gates, execution
receipts) lives in ``orze.skills`` since it is not SOP-specific.
"""
