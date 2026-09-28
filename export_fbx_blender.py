import sys
import argparse
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

MAP = {
    "Hips": "root",
    "Spine": "c_spine0",
    "Spine1": "c_spine1",
    "Spine2": "c_spine2",
    "Spine3": "c_spine3",
    "Neck": "c_neck",
    "Head": "c_head",
    "LeftShoulder": "l_clavicle",
    "LeftArm": "l_uparm",
    "LeftForeArm": "l_lowarm",
    "LeftHand": "l_wrist",
    "RightShoulder": "r_clavicle",
    "RightArm": "r_uparm",
    "RightForeArm": "r_lowarm",
    "RightHand": "r_wrist",
    "LeftUpLeg": "l_upleg",
    "LeftLeg": "l_lowleg",
    "LeftFoot": "l_foot",
    "LeftToeBase": "l_ball",
    "RightUpLeg": "r_upleg",
    "RightLeg": "r_lowleg",
    "RightFoot": "r_foot",
    "RightToeBase": "r_ball",
    "LeftHandThumb1": "l_thumb0",
    "LeftHandThumb2": "l_thumb1",
    "LeftHandThumb3": "l_thumb2",
    "LeftHandThumb4": "l_thumb3",
    "LeftHandIndex1": "l_index1",
    "LeftHandIndex2": "l_index2",
    "LeftHandIndex3": "l_index3",
    "LeftHandIndex4": "l_index_null",
    "LeftHandMiddle1": "l_middle1",
    "LeftHandMiddle2": "l_middle2",
    "LeftHandMiddle3": "l_middle3",
    "LeftHandMiddle4": "l_middle_null",
    "LeftHandRing1": "l_ring1",
    "LeftHandRing2": "l_ring2",
    "LeftHandRing3": "l_ring3",
    "LeftHandRing4": "l_ring_null",
    "LeftHandPinky1": "l_pinky0",
    "LeftHandPinky2": "l_pinky1",
    "LeftHandPinky3": "l_pinky2",
    "LeftHandPinky4": "l_pinky3",
    "RightHandThumb1": "r_thumb0",
    "RightHandThumb2": "r_thumb1",
    "RightHandThumb3": "r_thumb2",
    "RightHandThumb4": "r_thumb3",
    "RightHandIndex1": "r_index1",
    "RightHandIndex2": "r_index2",
    "RightHandIndex3": "r_index3",
    "RightHandIndex4": "r_index_null",
    "RightHandMiddle1": "r_middle1",
    "RightHandMiddle2": "r_middle2",
    "RightHandMiddle3": "r_middle3",
    "RightHandMiddle4": "r_middle_null",
    "RightHandRing1": "r_ring1",
    "RightHandRing2": "r_ring2",
    "RightHandRing3": "r_ring3",
    "RightHandRing4": "r_ring_null",
    "RightHandPinky1": "r_pinky0",
    "RightHandPinky2": "r_pinky1",
    "RightHandPinky3": "r_pinky2",
    "RightHandPinky4": "r_pinky3",
}


def basis(pos, left, right, root, head):
    x = (pos[right] - pos[left]).normalized()

    y = pos[head] - pos[root]
    y = (y - x * x.dot(y)).normalized()

    z = x.cross(y).normalized()
    y = z.cross(x).normalized()

    return Matrix((
        (x.x, y.x, z.x),
        (x.y, y.y, z.y),
        (x.z, y.z, z.z),
    ))


def load_source(skeleton):
    names = [str(x) for x in skeleton["joint_names"]]
    parents = skeleton["parents"].astype(int)
    offsets = skeleton["offsets"]
    pre = skeleton["pre_rotations"]

    idx = {name: i for i, name in enumerate(names)}

    rot = [None] * len(names)
    pos = [None] * len(names)

    for i in range(len(names)):
        q = pre[i]
        local = Quaternion((q[3], q[0], q[1], q[2])).to_matrix()

        p = parents[i]

        if p < 0:
            rot[i] = local
            pos[i] = Vector(offsets[i])
        else:
            rot[i] = rot[p] @ local
            pos[i] = pos[p] + rot[p] @ Vector(offsets[i])

    return idx, rot, pos


def rename_rig(arm):
    rename = {}

    for bone in list(arm.data.bones):
        old = bone.name

        if old.startswith("Monty_"):
            new = old[6:]
        elif old.startswith("mixamorig:"):
            new = old.split(":", 1)[1]
        else:
            continue

        rename[old] = new

    for old, new in rename.items():
        arm.data.bones[old].name = new

    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            for group in obj.vertex_groups:
                if group.name in rename:
                    group.name = rename[group.name]

    print(f"Renamed bones: {len(rename)}")


# def rename_rig(arm):
#     rename = {}
#
#     for bone in list(arm.data.bones):
#         if bone.name.startswith("Monty_"):
#             rename[bone.name] = bone.name[6:]
#
#     for old, new in rename.items():
#         arm.data.bones[old].name = new
#
#     for obj in bpy.context.scene.objects:
#         if obj.type == "MESH":
#             for group in obj.vertex_groups:
#                 if group.name in rename:
#                     group.name = rename[group.name]
#
#     if arm.name.startswith("Monty_"):
#         arm.name = arm.name[6:]


def parse_args():
    args = sys.argv[sys.argv.index("--") + 1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--skeleton", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    return parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args()

    bpy.ops.wm.read_factory_settings(use_empty=True)

    bpy.ops.import_scene.fbx(
        filepath=str(args.template),
        use_anim=False,
    )

    arm = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE")

    rename_rig(arm)

    required = [
        "Hips",
        "Head",
        "LeftUpLeg",
        "RightUpLeg",
        "LeftArm",
        "RightArm",
    ]

    missing = [name for name in required if name not in arm.data.bones]

    if missing:
        raise RuntimeError(f"Missing target bones: {missing}")

    arm.animation_data_clear()

    motion = np.load(args.motion)
    skeleton = np.load(args.skeleton)

    coords = motion["joint_coords"]
    rots = motion["joint_rots"]
    fps = float(motion["fps"])

    src_idx, src_rest_rot, src_rest_pos = load_source(skeleton)

    src_pos = {name: src_rest_pos[i] for name, i in src_idx.items()}

    bones = list(arm.data.bones)

    target_rest = {b.name: b.matrix_local.copy() for b in bones}

    target_rest_rot = {b.name: b.matrix_local.to_quaternion().to_matrix() for b in bones}

    target_rest_pos = {b.name: b.head_local.copy() for b in bones}

    target_parent = {b.name: b.parent.name if b.parent else None for b in bones}

    src_basis = basis(
        src_pos,
        "l_upleg",
        "r_upleg",
        "root",
        "c_head",
    )

    target_basis = basis(
        target_rest_pos,
        "LeftUpLeg",
        "RightUpLeg",
        "Hips",
        "Head",
    )

    align = (target_basis @ src_basis.transposed())

    src_height = (src_pos["c_head"] - src_pos["root"]).length

    target_height = (target_rest_pos["Head"] - target_rest_pos["Hips"]).length

    scale = target_height / src_height
    root_idx = src_idx["root"]

    mapped = {target: src_idx[source] for target, source in MAP.items() if target in target_rest and source in src_idx}

    src_ref_rot = {j: Matrix(rots[0, j].tolist()) for j in mapped.values()}

    depth = {}

    def get_depth(name):
        if name not in depth:
            parent = target_parent[name]

            depth[name] = (0 if parent is None else get_depth(parent) + 1)

        return depth[name]

    order = sorted(
        target_rest,
        key=get_depth,
    )

    arm.animation_data_create()
    arm.animation_data.action = bpy.data.actions.new("Motion")

    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.matrix_basis = Matrix.Identity(4)

    scene = bpy.context.scene

    scene.render.fps = round(fps)
    scene.frame_start = 1
    scene.frame_end = len(coords)

    previous_quat = {}

    for t in range(len(coords)):
        pose_rot = {}
        pose_pos = {}
        pose_matrix = {}

        root_delta = (align @ Vector(coords[t, root_idx] - coords[0, root_idx]) * scale)

        for name in order:
            parent = target_parent[name]

            if name in mapped:
                j = mapped[name]

                src_rot = Matrix(rots[t, j].tolist())

                delta = (src_rot @ src_ref_rot[j].transposed())

                delta = (align @ delta @ align.transposed())

                pose_rot[name] = (delta @ target_rest_rot[name])

            elif parent is None:
                pose_rot[name] = (target_rest_rot[name])

            else:
                pose_rot[name] = (pose_rot[parent] @ target_rest_rot[parent].transposed() @ target_rest_rot[name])

            if name == "Hips":
                pose_pos[name] = (target_rest_pos[name] + root_delta)

            elif parent is None:
                pose_pos[name] = (target_rest_pos[name])

            else:
                offset = (target_rest_pos[name] - target_rest_pos[parent])

                pose_pos[name] = (pose_pos[parent] + pose_rot[parent] @ target_rest_rot[parent].transposed() @ offset)

            mat = pose_rot[name].to_4x4()
            mat.translation = pose_pos[name]

            pose_matrix[name] = mat

        scene.frame_set(t + 1)

        for name in order:
            pb = arm.pose.bones.get(name)

            if pb is None:
                continue

            parent = target_parent[name]

            if parent is None:
                mat_basis = (target_rest[name].inverted() @ pose_matrix[name])

            else:
                mat_basis = (
                    target_rest[name].inverted() @ target_rest[parent] @ pose_matrix[parent].inverted() @ pose_matrix[name]
                )

            loc, quat, _ = mat_basis.decompose()

            if (t == 0 and name in [
                "LeftShoulder",
                "LeftArm",
                "LeftForeArm",
                "RightShoulder",
                "RightArm",
                "RightForeArm",
            ]):
                print(
                    f"{name:16s} "
                    f"loc=("
                    f"{loc.x:+.6f}, "
                    f"{loc.y:+.6f}, "
                    f"{loc.z:+.6f}) "
                    f"angle="
                    f"{np.degrees(quat.angle):.6f}"
                )

            if name in previous_quat:
                quat.make_compatible(previous_quat[name])

            previous_quat[name] = (quat.copy())

            pb.rotation_quaternion = quat
            pb.scale = (1.0, 1.0, 1.0)

            if name == "Hips":
                pb.location = loc
            else:
                pb.location = (
                    0.0,
                    0.0,
                    0.0,
                )

            if name in mapped:
                pb.keyframe_insert(
                    data_path="rotation_quaternion",
                    frame=t + 1,
                    group=name,
                )

            if name == "Hips":
                pb.keyframe_insert(
                    data_path="location",
                    frame=t + 1,
                    group=name,
                )

        bpy.context.view_layer.update()

    action = arm.animation_data.action

    if action is None:
        raise RuntimeError("No animation action created")

    curve_count = len(action.fcurves)
    key_count = 0

    for curve in action.fcurves:
        for key in curve.keyframe_points:
            key.interpolation = "LINEAR"
            key_count += 1

    scene.frame_set(1)
    bpy.context.view_layer.update()

    print("\nFrame 1 pose check:")

    for name in [
        "LeftShoulder",
        "LeftArm",
        "LeftForeArm",
        "RightShoulder",
        "RightArm",
        "RightForeArm",
    ]:
        pb = arm.pose.bones.get(name)

        if pb is None:
            continue

        loc, quat, _ = (pb.matrix_basis.decompose())

        print(
            f"{name:16s} "
            f"loc=("
            f"{loc.x:+.6f}, "
            f"{loc.y:+.6f}, "
            f"{loc.z:+.6f}) "
            f"angle="
            f"{np.degrees(quat.angle):.6f}"
        )

    print()
    print(f"Frames: {len(coords)}")
    print(f"Mapped bones: {len(mapped)}")
    print(f"Animation curves: {curve_count}")
    print(f"Keyframes: {key_count}")

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    bpy.ops.export_scene.fbx(
        filepath=str(args.output),
        use_selection=False,
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        bake_anim_simplify_factor=0.0,
    )

    print(f"Saved: {args.output}")
