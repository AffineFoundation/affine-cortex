from click.testing import CliRunner

from affine.cli.main import cli
from affine.database.dao.inference_endpoints import InferenceEndpointsDAO


def test_db_delete_endpoint_command_is_not_exposed():
    result = CliRunner().invoke(
        cli,
        ["db", "delete-endpoint", "--name", "lium-b200-temp-2"],
    )

    assert result.exit_code == 2
    assert "No such command 'delete-endpoint'" in result.output


def test_endpoint_dao_has_no_name_based_hard_delete():
    assert "delete" not in InferenceEndpointsDAO.__dict__
