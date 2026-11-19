HEAD = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>
:root{{--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#898781;--surface-1:#f3f2ee;--surface-2:#fff;--border:rgba(11,11,11,.10)}}
@media (prefers-color-scheme:dark){{:root{{--text-primary:#f0efec;--text-secondary:#c3c2b7;--text-muted:#898781;--surface-1:#232321;--surface-2:#1a1a19;--border:rgba(255,255,255,.12)}}body{{background:#1a1a19}}}}
body{{font-family:system-ui,sans-serif;color:var(--text-primary);max-width:760px;margin:1rem auto;padding:0 12px}}
.legend{{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:12px;color:var(--text-secondary);margin:.5rem 0 6px}}
.legend span{{display:flex;align-items:center;gap:5px}}
</style></head><body><h1 style="font-size:18px;font-weight:500">{title}</h1>
<p style="font-size:13px;color:var(--text-secondary)">{note}</p>
"""
