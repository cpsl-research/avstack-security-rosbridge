from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from avstack.datastructs import DataContainer
    from avstack.geometry import ReferenceFrame, Shape

from avstack.config import HOOKS
from avstack_rosbag import RosbagHook


@HOOKS.register_module()
class AdversaryRosbagHook(RosbagHook):
    def __call__(
        self,
        detections: "DataContainer",
        field_of_view: "Shape",
        reference: "ReferenceFrame",
        agent_name: str,
        sensor_name: str,
        logger=None,
        *args,
        **kwargs,
    ):
        # run the model straight up
        detections, field_of_view, attacked_agents = self.hook(
            detections=detections,
            field_of_view=field_of_view,
            reference=reference,
            agent_name=agent_name,
            sensor_name=sensor_name,
            logger=logger,
        )
        return detections, field_of_view, attacked_agents
