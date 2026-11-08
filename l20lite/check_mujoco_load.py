#!/usr/bin/env python3
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parent


def show_model(path: Path):

    print(
        f"\n=== Loading "
        f"{path} ==="
    )

    model = (
        mujoco.MjModel
        .from_xml_path(
            str(path)
        )
    )

    print(
        f"OK: "
        f"nq={model.nq}, "
        f"nv={model.nv}, "
        f"nu={model.nu}, "
        f"njnt={model.njnt}, "
        f"nbody={model.nbody}, "
        f"ngeom={model.ngeom}"
    )

    for jid in range(
        model.njnt
    ):

        name = (
            mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                jid,
            )
        )

        if name:

            lo, hi = (
                model.jnt_range[jid]
            )

            print(
                f"{jid:2d}: "
                f"{name:<32s} "
                f"range="
                f"[{lo:.4f}, "
                f"{hi:.4f}]"
            )


def main():

    dual = (
        ROOT
        / "generated"
        / "dual_l20lite.urdf"
    )

    if not dual.exists():

        raise FileNotFoundError(
            f"{dual} does not exist. "
            "Run "
            "prepare_l20lite_urdf.py "
            "first."
        )

    show_model(
        dual
    )


if __name__ == "__main__":
    main()
