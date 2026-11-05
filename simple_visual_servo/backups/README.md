# Backups

- `waterloo_steel_full.urdf.bak`: exact original thesis robot URDF, including its
  original absolute mesh paths. This is a reference backup, not a portable runtime file.
- `waterloo_steel_full.portable.urdf.bak`: backup of the working robot URDF with
  mesh paths relative to `assets/`.
- `blue_block.urdf.bak`, `ground.urdf.bak`: backups of the two simple scene URDFs.
- `simple_visual_servo.py.bak`: backup of the completed Python demo.
- `simple_visual_servo.stationary-default.py.bak`: previous version before the
  indefinite moving-block default, mobility scaling, and scene trails were added.

To restore a working file, copy its backup to the matching location in the
parent folder or `assets/`. The portable robot URDF uses the shared mesh files
in `assets/meshes/`. The simulation loads only working files, never these backups.
Original mesh assets are copied unchanged and do not need a duplicate backup.
