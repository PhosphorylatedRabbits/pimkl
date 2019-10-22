def is_sequence(arg):
    return (
        not hasattr(arg, "strip") and
        not hasattr(arg, "shape") and
        (hasattr(arg, "__getitem__") or hasattr(arg, "__iter__"))
    )


def is_sequence_of_sequence(arg):
    if is_sequence(arg):
        try:
            return is_sequence(next(arg.__iter__()))
        except Exception:
            return False
    else:
        return False
