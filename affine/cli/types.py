"""
Custom Click parameter types for CLI commands.
"""

import click


class UIDParamType(click.ParamType):
    """Custom parameter type for UID that supports negative values.

    Accepts:
    - Regular integers: 42, 0, 100
    - Negative prefix 'n': n1 -> -1, n2 -> -2

    Examples:
        @click.argument("uid", type=UID)
        def my_command(uid):
            ...

        $ my_command 42    # uid = 42
        $ my_command n1    # uid = -1
        $ my_command n10   # uid = -10
    """
    name = "uid"

    def convert(self, value, param, ctx):
        if value is None:
            return None

        if isinstance(value, int):
            return value

        value_str = str(value).strip()

        # Handle 'n' prefix for negative numbers (e.g., n1 -> -1)
        if value_str.lower().startswith('n') and len(value_str) > 1:
            try:
                return -int(value_str[1:])
            except ValueError:
                self.fail(
                    f"'{value}' is not a valid UID. Use 'n1' for -1 or regular integers.",
                    param,
                    ctx
                )

        # Regular integer
        try:
            return int(value_str)
        except ValueError:
            self.fail(f"'{value}' is not a valid UID.", param, ctx)


# Singleton instance for convenience
UID = UIDParamType()
