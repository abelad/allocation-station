"""FIX Protocol Support for trading integration."""

from typing import Dict, Optional
from datetime import datetime
from enum import Enum


class FIXMsgType(Enum):
    """FIX message types."""
    LOGON = '0'
    HEARTBEAT = '1'
    NEW_ORDER_SINGLE = 'D'
    ORDER_CANCEL_REQUEST = 'F'
    EXECUTION_REPORT = '8'
    MARKET_DATA_REQUEST = 'V'


class FIXProtocolHandler:
    """Handle FIX protocol messages for trading integration."""

    def __init__(self, sender_comp_id: str, target_comp_id: str):
        """Initialize FIX protocol handler."""
        self.sender_comp_id = sender_comp_id
        self.target_comp_id = target_comp_id
        self.msg_seq_num = 1

    def create_logon_message(self) -> str:
        """Create FIX logon message."""
        msg = self._create_header(FIXMsgType.LOGON)
        msg += f"108=30|"  # HeartBtInt
        msg += f"98=0|"   # EncryptMethod
        return self._add_checksum(msg)

    def create_new_order(self, symbol: str, side: str, quantity: int,
                        price: Optional[float] = None) -> str:
        """Create FIX new order message."""
        msg = self._create_header(FIXMsgType.NEW_ORDER_SINGLE)
        msg += f"11={self._generate_order_id()}|"  # ClOrdID
        msg += f"21=1|"   # HandlInst
        msg += f"55={symbol}|"  # Symbol
        msg += f"54={side}|"    # Side (1=Buy, 2=Sell)
        msg += f"60={self._get_timestamp()}|"  # TransactTime
        msg += f"38={quantity}|"  # OrderQty

        if price:
            msg += f"40=2|"  # OrdType (2=Limit)
            msg += f"44={price}|"  # Price
        else:
            msg += f"40=1|"  # OrdType (1=Market)

        return self._add_checksum(msg)

    def parse_execution_report(self, fix_msg: str) -> Dict:
        """Parse FIX execution report."""
        fields = self._parse_message(fix_msg)
        return {
            'order_id': fields.get('11'),
            'exec_id': fields.get('17'),
            'ord_status': fields.get('39'),
            'symbol': fields.get('55'),
            'side': fields.get('54'),
            'cum_qty': fields.get('14'),
            'avg_px': fields.get('6')
        }

    def _create_header(self, msg_type: FIXMsgType) -> str:
        """Create FIX message header."""
        msg = f"8=FIX.4.4|"  # BeginString
        msg += f"9=0|"  # BodyLength (calculated later)
        msg += f"35={msg_type.value}|"  # MsgType
        msg += f"49={self.sender_comp_id}|"  # SenderCompID
        msg += f"56={self.target_comp_id}|"  # TargetCompID
        msg += f"34={self.msg_seq_num}|"  # MsgSeqNum
        msg += f"52={self._get_timestamp()}|"  # SendingTime
        self.msg_seq_num += 1
        return msg

    def _add_checksum(self, msg: str) -> str:
        """Add FIX checksum."""
        checksum = sum(ord(c) for c in msg) % 256
        return msg + f"10={checksum:03d}|"

    def _get_timestamp(self) -> str:
        """Get FIX timestamp format."""
        return datetime.utcnow().strftime('%Y%m%d-%H:%M:%S.%f')[:-3]

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        return f"ORD{datetime.now().strftime('%Y%m%d%H%M%S')}"

    def _parse_message(self, fix_msg: str) -> Dict:
        """Parse FIX message into field dictionary."""
        fields = {}
        for field in fix_msg.split('|'):
            if '=' in field:
                tag, value = field.split('=', 1)
                fields[tag] = value
        return fields
