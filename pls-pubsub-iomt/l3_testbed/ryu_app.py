"""Ryu OpenFlow 1.3 learning switch for the testbed (s1 = AP switch, s2 = core switch).

Forwarding only; the orchestration policy runs in the orchestrator (l3_testbed/orchestrator.py),
which plays the SDN application role. Port statistics are logged every 10 s for the paper's
system description and for debugging.
"""
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.lib.packet import ethernet, ether_types, packet
from ryu.ofproto import ofproto_v1_3


class IoMTSwitch(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.datapaths = {}
        self.monitor = hub.spawn(self._monitor)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features(self, ev):
        dp = ev.msg.datapath
        self.datapaths[dp.id] = dp
        ofp, parser = dp.ofproto, dp.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        self.add_flow(dp, 0, match, actions)

    def add_flow(self, dp, priority, match, actions, buffer_id=None):
        ofp, parser = dp.ofproto, dp.ofproto_parser
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        kw = dict(datapath=dp, priority=priority, match=match, instructions=inst)
        if buffer_id is not None and buffer_id != ofp.OFP_NO_BUFFER:
            kw["buffer_id"] = buffer_id
        dp.send_msg(parser.OFPFlowMod(**kw))

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in(self, ev):
        msg = ev.msg
        dp = msg.datapath
        ofp, parser = dp.ofproto, dp.ofproto_parser
        in_port = msg.match["in_port"]
        eth = packet.Packet(msg.data).get_protocols(ethernet.ethernet)[0]
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return
        table = self.mac_to_port.setdefault(dp.id, {})
        table[eth.src] = in_port
        out_port = table.get(eth.dst, ofp.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]
        if out_port != ofp.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=eth.dst, eth_src=eth.src)
            if msg.buffer_id != ofp.OFP_NO_BUFFER:
                self.add_flow(dp, 1, match, actions, msg.buffer_id)
                return
            self.add_flow(dp, 1, match, actions)
        data = msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None
        dp.send_msg(parser.OFPPacketOut(datapath=dp, buffer_id=msg.buffer_id, in_port=in_port,
                                        actions=actions, data=data))

    def _monitor(self):
        while True:
            for dp in list(self.datapaths.values()):
                dp.send_msg(dp.ofproto_parser.OFPPortStatsRequest(dp, 0, dp.ofproto.OFPP_ANY))
            hub.sleep(10)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats(self, ev):
        for st in ev.msg.body:
            if st.port_no < 0xFFFFFF00:
                self.logger.info("dp %s port %s rx_pkts %d tx_pkts %d tx_dropped %d", ev.msg.datapath.id,
                                 st.port_no, st.rx_packets, st.tx_packets, st.tx_dropped)
