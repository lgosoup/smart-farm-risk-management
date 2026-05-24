"""Bandit + optional LLM gate improvement module.

This module is intentionally separated from the real-time fuzzy inference.
It updates *baseline* calibration (membership points + rule weights) in small steps.
"""
