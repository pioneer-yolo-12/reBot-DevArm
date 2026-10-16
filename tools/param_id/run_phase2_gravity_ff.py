#!/usr/bin/env python3
"""Compatibility entry point for the unified raw-event acquisition pipeline.

Use --mode tracking for gain experiments; --mode identify for repeated sweeps.
C/E are diagnostics computed offline by analyze_friction.py.
"""
from run_phase2_pointmatch import main

if __name__ == '__main__':
    main()
