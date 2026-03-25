#!/usr/bin/env python3

import os
import sys


def usage():
    raise SystemExit(
        "usage: path_helper.py <expand|canonicalize|ensure-within|relative> ..."
    )


def common_root(root_dir, candidate_path):
    try:
        return os.path.commonpath([root_dir, candidate_path]) == root_dir
    except ValueError:
        return False


def ensure_within(root_dir, candidate_path, label):
    root_dir = os.path.realpath(root_dir)
    candidate_path = os.path.realpath(candidate_path)

    if not common_root(root_dir, candidate_path):
        raise SystemExit(
            "Error: {label} '{candidate}' must be located under install root '{root}'".format(
                label=label,
                candidate=candidate_path,
                root=root_dir,
            )
        )

    return root_dir, candidate_path


def cmd_expand(args):
    if len(args) != 6:
        usage()

    template, base_dir, project_root, profile_id, backend, install_root = args
    for src, dst in (
        ("{project_root}", project_root),
        ("{profile}", profile_id),
        ("{backend}", backend),
        ("{root}", install_root),
    ):
        template = template.replace(src, dst)

    if os.path.isabs(template):
        print(os.path.normpath(template))
    else:
        print(os.path.normpath(os.path.join(base_dir, template)))


def cmd_canonicalize(args):
    if len(args) != 1:
        usage()
    print(os.path.realpath(args[0]))


def cmd_ensure_within(args):
    if len(args) != 3:
        usage()
    ensure_within(args[0], args[1], args[2])


def cmd_relative(args):
    if len(args) != 2:
        usage()

    root_dir, candidate_path = ensure_within(args[0], args[1], "path")
    print(os.path.relpath(candidate_path, root_dir))


def main():
    if len(sys.argv) < 2:
        usage()

    command = sys.argv[1]
    args = sys.argv[2:]

    if command == "expand":
        cmd_expand(args)
    elif command == "canonicalize":
        cmd_canonicalize(args)
    elif command == "ensure-within":
        cmd_ensure_within(args)
    elif command == "relative":
        cmd_relative(args)
    else:
        usage()


if __name__ == "__main__":
    main()
