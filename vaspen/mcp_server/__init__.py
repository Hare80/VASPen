"""Headless MCP (Model Context Protocol) server for VASPen.

Exposes the core layer (``vaspen.core``) as MCP tools over stdio so AI
clients can open, analyze, transform and generate VASP inputs for
structures without the GUI. See CLAUDE.md §7.11 for the settled design.
"""
