"""
System Config DAO

Manages dynamic configuration parameters.
"""

from typing import Dict, Any, List, Optional
from affine.database.base_dao import BaseDAO
from affine.database.schema import get_table_name


class SystemConfigDAO(BaseDAO):
    """DAO for system_config table.
    
    Stores dynamic configuration parameters.
    PK: CONFIG
    SK: PARAM#{param_name}
    """
    
    def __init__(self):
        self.table_name = get_table_name("system_config")
        super().__init__()
    
    def _make_pk(self) -> str:
        """Generate partition key."""
        return "CONFIG"
    
    def _make_sk(self, param_name: str) -> str:
        """Generate sort key."""
        return f"PARAM#{param_name}"
    
    async def set_param(
        self,
        param_name: str,
        param_value: Any,
        param_type: str,
        description: str = "",
        updated_by: str = "system"
    ) -> Dict[str, Any]:
        """Set a configuration parameter.
        
        Args:
            param_name: Parameter name
            param_value: Parameter value
            param_type: Parameter type (str/int/float/bool/dict/list)
            description: Parameter description
            updated_by: Who updated the parameter
            
        Returns:
            Saved config item
        """
        import time
        
        # Get existing config to increment version
        existing = await self.get_param(param_name)
        version = (existing.get('version', 0) + 1) if existing else 1
        
        item = {
            'pk': self._make_pk(),
            'sk': self._make_sk(param_name),
            'param_name': param_name,
            'param_value': param_value,
            'param_type': param_type,
            'description': description,
            'updated_at': int(time.time()),
            'updated_by': updated_by,
            'version': version,
        }
        
        return await self.put(item)
    
    async def get_param(self, param_name: str) -> Optional[Dict[str, Any]]:
        """Get a configuration parameter.
        
        Args:
            param_name: Parameter name
            
        Returns:
            Config item if found, None otherwise
        """
        pk = self._make_pk()
        sk = self._make_sk(param_name)
        
        return await self.get(pk, sk)
    
    async def get_param_value(self, param_name: str, default: Any = None) -> Any:
        """Get parameter value directly.
        
        Args:
            param_name: Parameter name
            default: Default value if not found
            
        Returns:
            Parameter value or default
        """
        param = await self.get_param(param_name)
        return param.get('param_value', default) if param else default
    
    async def get_all_params(self) -> Dict[str, Any]:
        """Get all configuration parameters.
        
        Returns:
            Dictionary mapping param names to values
        """
        pk = self._make_pk()
        
        items = await self.query(pk=pk)
        
        return {item['param_name']: item['param_value'] for item in items}
    
    async def delete_param(self, param_name: str) -> bool:
        """Delete a configuration parameter.
        
        Args:
            param_name: Parameter name
            
        Returns:
            True if deleted successfully
        """
        pk = self._make_pk()
        sk = self._make_sk(param_name)
        
        return await self.delete(pk, sk)
    
    async def list_all_configs(self) -> List[Dict[str, Any]]:
        """List all configuration parameters with metadata.
        
        Returns:
            List of config items with full details
        """
        pk = self._make_pk()
        
        return await self.query(pk=pk)
    
    # Environment Configuration Management
    #
    # Environment rows use two independent gates:
    #   enabled_for_sampling controls task-id materialization/executor work.
    #   enabled_for_scoring controls DECIDE/rank score columns.
    # The legacy ``enabled`` key is accepted by StateStore as a sampling alias.

    # Blacklist Configuration Management
    
    async def get_blacklist(self) -> List[str]:
        """Get blacklisted hotkeys from database.
        
        Returns:
            List of blacklisted hotkey strings
        """
        blacklist = await self.get_param_value('miner_blacklist', default=[])
        return blacklist if isinstance(blacklist, list) else []
    
    async def set_blacklist(
        self, hotkeys: List[str], updated_by: str = "system"
    ) -> Dict[str, Any]:
        """Set blacklisted hotkeys.
        
        Args:
            hotkeys: List of hotkey strings to blacklist
            updated_by: Who updated the parameter
            
        Returns:
            Saved config item
        """
        # Remove duplicates and empty strings
        unique_hotkeys = list(set(hk.strip() for hk in hotkeys if hk.strip()))
        
        return await self.set_param(
            param_name='miner_blacklist',
            param_value=unique_hotkeys,
            param_type='list',
            description='Blacklisted miner hotkeys',
            updated_by=updated_by
        )
    
    async def add_to_blacklist(
        self, hotkeys: List[str], updated_by: str = "system"
    ) -> Dict[str, Any]:
        """Add hotkeys to blacklist.
        
        Args:
            hotkeys: List of hotkey strings to add
            updated_by: Who updated the parameter
            
        Returns:
            Saved config item
        """
        current_blacklist = await self.get_blacklist()
        updated_blacklist = list(set(current_blacklist + hotkeys))
        return await self.set_blacklist(updated_blacklist, updated_by)
    
    async def remove_from_blacklist(
        self, hotkeys: List[str], updated_by: str = "system"
    ) -> Dict[str, Any]:
        """Remove hotkeys from blacklist.

        Args:
            hotkeys: List of hotkey strings to remove
            updated_by: Who updated the parameter

        Returns:
            Saved config item
        """
        current_blacklist = await self.get_blacklist()
        updated_blacklist = [hk for hk in current_blacklist if hk not in hotkeys]
        return await self.set_blacklist(updated_blacklist, updated_by)

    # System Miners Configuration Management

    async def get_system_miners(self) -> Dict[str, Dict]:
        """Get all system miners configuration.

        System miners are benchmark models (like GPT-4o, Claude) that participate
        in scoring but don't receive actual rewards. They use UIDs > 1000.

        Storage format (param_name="system_miners"):
        {
            "1001": {"model": "openai/gpt-4o"},
            "1002": {"model": "anthropic/claude-3.5-sonnet"}
        }

        Returns:
            Dict mapping uid string to config dict
        """
        return await self.get_param_value("system_miners", default={})

    async def set_system_miner(
        self, uid: int, model: str, updated_by: str = "cli"
    ) -> Dict[str, Any]:
        """Set a system miner configuration.

        System miners use UIDs > 1000 to distinguish them from
        regular miners on the metagraph.

        Args:
            uid: System miner UID (must be > 1000)
            model: Model identifier (e.g., "openai/gpt-4o")
            updated_by: Who updated the parameter

        Returns:
            Saved config item

        Raises:
            ValueError: If uid is not > 1000
        """
        if uid <= 1000:
            raise ValueError("System miner UID must be > 1000")

        current = await self.get_system_miners()
        current[str(uid)] = {"model": model}

        return await self.set_param(
            param_name="system_miners",
            param_value=current,
            param_type="dict",
            description="System miners configuration",
            updated_by=updated_by
        )

    async def delete_system_miner(
        self, uid: int, updated_by: str = "cli"
    ) -> bool:
        """Delete a system miner configuration.

        Args:
            uid: System miner UID to delete
            updated_by: Who updated the parameter

        Returns:
            True if deleted, False if not found
        """
        current = await self.get_system_miners()
        if str(uid) in current:
            del current[str(uid)]
            await self.set_param(
                param_name="system_miners",
                param_value=current,
                param_type="dict",
                description="System miners configuration",
                updated_by=updated_by
            )
            return True
        return False
