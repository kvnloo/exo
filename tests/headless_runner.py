import multiprocessing as mp
import socket
import time

import anyio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from hypercorn import Config
from hypercorn.asyncio import serve  # pyright: ignore[reportUnknownVariableType]
from loguru import logger
from pydantic import BaseModel

from exo.shared.constants import EXO_MODELS_DIR
from exo.shared.logging import InterceptLogger, logger_setup
from exo.shared.models.model_cards import MODEL_CARDS, ModelId
from exo.shared.types.api import ChatCompletionMessage, ChatCompletionTaskParams
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.commands import CommandId
from exo.shared.types.common import Host, NodeId
from exo.shared.types.events import ChunkGenerated, Event, RunnerStatusUpdated
from exo.shared.types.tasks import (
    ChatCompletion,
    ConnectToGroup,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
)
from exo.shared.types.worker.instances import (
    BoundInstance,
    Instance,
    InstanceId,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerShutdown,
    ShardAssignments,
)
from exo.shared.types.worker.shards import PipelineShardMetadata, TensorShardMetadata
from exo.utils.channels import MpReceiver, MpSender, channel, mp_channel
from exo.utils.info_gatherer.info_gatherer import GatheredInfo, InfoGatherer
from exo.worker.runner.bootstrap import entrypoint


class Tests(BaseModel):
    # list[hostname, ip addr]
    devs: list[list[str]]
    model_id: ModelId


mp.set_start_method("spawn", force=True)
logger_setup(None)


async def main():
    logger.info("starting cool server majig")
    cfg = Config()
    cfg.bind = "0.0.0.0:8000"
    # nb: shared.logging needs updating if any of this changes
    cfg.accesslog = "-"
    cfg.errorlog = "-"
    cfg.logger_class = InterceptLogger
    app = FastAPI()
    app.post("/run_test")(run_test)
    app.get("/tb_detection")(tb_detection)
    app.get("/models")(list_models)
    await serve(
        app,  # type: ignore
        cfg,
    )


async def tb_detection():
    send, recv = channel[GatheredInfo]()
    ig = InfoGatherer(send)
    with anyio.move_on_after(1):
        await ig._monitor_system_profiler_thunderbolt_data()  # pyright: ignore[reportPrivateUsage]
    with recv:
        return recv.collect()


def list_models():
    sent = set[str]()
    for path in EXO_MODELS_DIR.rglob("model-*.safetensors"):
        if "--" not in path.parent.name:
            continue
        name = path.parent.name.replace("--", "/")
        if name in sent:
            continue
        sent.add(name)
        yield ModelId(path.parent.name.replace("--", "/"))


async def run_test(test: Tests):
    iid = InstanceId((str(test.devs)))
    weird_hn = socket.gethostname()
    for dev in test.devs:
        if weird_hn.startswith(dev[0]) or dev[0].startswith(weird_hn):
            hn = dev[0]
            break
    else:
        raise ValueError(f"{weird_hn} not in {test.devs}")

    async def run():
        logger.info(f"testing {test.model_id}")

        for instance in (
            ring_instance(test, iid, hn),
            jaccl_instance(test, iid),
        ):
            if instance is None:
                continue
            recv = await execute_test(test, instance, hn)

            str_out = ""

            with recv:
                try:
                    async for item in recv:
                        if isinstance(item, ChunkGenerated):
                            assert isinstance(item.chunk, TokenChunk)
                            str_out += item.chunk.text

                        if isinstance(item, RunnerStatusUpdated) and isinstance(
                            item.runner_status, (RunnerFailed, RunnerShutdown)
                        ):
                            yield str_out + "\n"
                            yield item.model_dump_json() + "\n"
                except anyio.ClosedResourceError:
                    pass

    return StreamingResponse(run())


def ring_instance(test: Tests, iid: InstanceId, hn: str) -> Instance | None:
    hbn = [Host(ip="i dont care", port=52416) for _ in test.devs]
    world_size = len(test.devs)
    for i in range(world_size):
        if test.devs[i][0] == hn:
            hn = test.devs[i][0]
            if i - 1 >= 0:
                hbn[i - 1] = Host(ip=test.devs[i - 1][1], port=52416)
            if i + 1 < len(test.devs):
                hbn[i + 1] = Host(ip=test.devs[i + 1][1], port=52416)
            hbn[i] = Host(ip="0.0.0.0", port=52416)
            break
    else:
        raise ValueError(f"{hn} not in {test.devs}")

    card = next(
        (card for card in MODEL_CARDS.values() if card.model_id == test.model_id), None
    )
    if card is None:
        return None
    instance = MlxRingInstance(
        instance_id=iid,
        ephemeral_port=52416,
        hosts_by_node={NodeId(hn): hbn},
        shard_assignments=ShardAssignments(
            model_id=test.model_id,
            node_to_runner={NodeId(host[0]): RunnerId(host[0]) for host in test.devs},
            runner_to_shard={
                RunnerId(test.devs[i][0]): PipelineShardMetadata(
                    model_card=card,
                    device_rank=i,
                    world_size=world_size,
                    start_layer=(card.n_layers // world_size) * i,
                    end_layer=min(
                        card.n_layers, (card.n_layers // world_size) * (i + 1)
                    ),
                    n_layers=min(card.n_layers, (card.n_layers // world_size) * (i + 1))
                    - (card.n_layers // world_size) * i,
                )
                for i in range(world_size)
            },
        ),
    )

    return instance


async def execute_test(test: Tests, instance: Instance, hn: str) -> MpReceiver[Event]:
    world_size = len(test.devs)
    iid = InstanceId(str(test.devs))
    _handle, recv, send = new_runner(instance, hn)
    if world_size > 1:
        send.send(ConnectToGroup(instance_id=iid))
    send.send(LoadModel(instance_id=iid))
    send.send(StartWarmup(instance_id=iid))
    send.send(
        ChatCompletion(
            task_params=ChatCompletionTaskParams(
                model=test.model_id,
                messages=[
                    ChatCompletionMessage(
                        role="system", content="You are a helpful assistant"
                    ),
                    ChatCompletionMessage(
                        role="user", content="What is the capital of France?"
                    ),
                ],
                max_tokens=50,
            ),
            command_id=CommandId("yo"),
            instance_id=iid,
        )
    )
    send.send(Shutdown(runner_id=RunnerId(hn), instance_id=iid))

    return recv


def jaccl_instance(test: Tests, iid: InstanceId) -> MlxJacclInstance | None:
    card = next(
        (card for card in MODEL_CARDS.values() if card.model_id == test.model_id), None
    )
    if card is None:
        return None
    world_size = len(test.devs)

    return MlxJacclInstance(
        instance_id=iid,
        jaccl_devices=[[None, "rdma_en3"], ["rdma_en3", None]],
        # rank 0 is always coordinator
        jaccl_coordinators={
            NodeId(host[0]): test.devs[0][1] + ":52416" for host in test.devs
        },
        shard_assignments=ShardAssignments(
            model_id=test.model_id,
            node_to_runner={NodeId(host[0]): RunnerId(host[0]) for host in test.devs},
            runner_to_shard={
                RunnerId(host[0]): TensorShardMetadata(
                    model_card=card,
                    device_rank=i,
                    world_size=world_size,
                    start_layer=0,
                    end_layer=card.n_layers,
                    n_layers=card.n_layers,
                )
                for i, host in enumerate(test.devs)
            },
        ),
    )


def new_runner(
    instance: Instance,
    hn: str,
) -> tuple[mp.Process, MpReceiver[Event], MpSender[Task]]:
    bound_instance = BoundInstance(
        instance=instance, bound_runner_id=RunnerId(hn), bound_node_id=NodeId(hn)
    )
    ev_send, ev_recv = mp_channel[Event]()
    task_send, task_recv = mp_channel[Task]()

    runner_process = mp.Process(
        target=entrypoint,
        args=(
            bound_instance,
            ev_send,
            task_recv,
            logger,
        ),
    )
    runner_process._pls_dont_gc = (ev_send, task_recv)  # type: ignore
    runner_process.start()
    time.sleep(0.1)
    return (runner_process, ev_recv, task_send)


if __name__ == "__main__":
    anyio.run(main)
