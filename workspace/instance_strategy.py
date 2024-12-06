from abc import ABC, abstractmethod
import logging
import os
import json
import shutil
from backend.server import Server
from frontend.client_factory import ClientFactory
from sd_utils import Results, Region, compute_regions_size, merge_boxes_in_results, cleanup, extract_images_from_video, read_results_dict
from streamduet_utils import list_frames,get_images_length,get_image_extension
from workspace.base_instance_strategy import InstanceStrategy
from workspace.streamduet_RoI_strategy import StreamDuetRoIStrategy
import time
class GTStrategy(InstanceStrategy):
    def run(self, args):
        self.logger.warning(f"Running GT in mode on {args.video_name}")
        if args.mode == "emulation":
            self.server = Server(self.config)
            self.client = ClientFactory.get_client(self.config, args.client_id, server=self.server)
        else:
            self.client = ClientFactory.get_client(self.config, args.client_id, hname=args.hname)

        results, bw, lt = self.analyze_video_mpeg(args.video_name, args.high_images_path, args.enforce_iframes)
        return results, bw, lt

class MPEGStrategy(InstanceStrategy):
    def run(self, args):
        self.logger.warning(f"Running in MPEG mode with resolution {args.low_resolution} on {args.video_name}")
        if args.mode == "emulation":
            self.server = Server(self.config)
            self.client = ClientFactory.get_client(self.config, args.client_id, server=self.server)
        else:
            self.client = ClientFactory.get_client(self.config, args.client_id, hname=args.hname)

        results, bw , lt = self.analyze_video_mpeg(args.video_name, args.high_images_path, args.enforce_iframes)
        return results, bw, lt


class StrategyFactory:
    @staticmethod
    def get_strategy(args, config, logger):
        if args.method == 'gt':
            return GTStrategy(config, logger)
        elif args.method == 'mpeg':
            return MPEGStrategy(config, logger)
        elif args.method == 'streamduetRoI':
            return StreamDuetRoIStrategy(config, logger)
        else:
            raise ValueError(f"Unknown method {args.method}")