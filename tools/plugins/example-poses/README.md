# Example camera placement plugin

A working plugin to copy when writing your own (see `docs/PLUGINS.md`). It
places the cameras with COLMAP's incremental mapper, run from its own
script, so it gives about what the built-in camera placement gives; what
matters is the shape of it:

- `ez2d-plugin.toml`: what it provides, the command with its placeholders,
  the progress pattern, and a `[[license]]` for each part;
- `run.py`: reads the photos and masks, writes `sparse/0`;
- `LICENSE.txt`: shown to the user before the plugin can be used.

Try it:

```sh
ez2d plugins install tools/plugins/example-poses
ez2d plugins accept example-poses
ez2d plugins use poses example-poses
ez2d run ~/scans/skull
ez2d plugins use poses built-in
```

It needs `python3` on `PATH` (on Windows, change it to `python` in the
manifest); it runs the COLMAP the app uses.
