import sys
import bpy

path = sys.argv[sys.argv.index("--") + 1]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=path, use_anim=False)

arm = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE")

for b in arm.data.bones:
    d = (b.tail_local - b.head_local).normalized()

    print(
        f"{b.name:30s} "
        f"parent={b.parent.name if b.parent else '-':25s} "
        f"connect={str(b.use_connect):5s} "
        f"deform={str(b.use_deform):5s} "
        f"dir=({d.x:+.2f}, {d.y:+.2f}, {d.z:+.2f})"
    )
