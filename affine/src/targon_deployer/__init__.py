"""Targon Deployer Service.

Reconciles the set of Targon deployments against a target (stage-1: the
current champion). Writes deployment metadata to DynamoDB so the
ProviderRouter can read it synchronously on every task fetch.
"""

from .service import TargonDeployerService

__all__ = ["TargonDeployerService"]
