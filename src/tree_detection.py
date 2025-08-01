import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
import json
import copy
import io
import requests
import time
import re
from datetime import datetime, timedelta


class VideoTimestampGenerator:
    """Helper class for generating millisecond precision timestamps based on video start time"""

    def __init__(self, video_start_time=None, fps=30.0):
        """
        Initialize with video start time

        Args:
            video_start_time: Can be:
                - datetime object
                - ISO string like "2025-07-07T10:30:00.123"
                - Unix timestamp in seconds
                - Unix timestamp in milliseconds (13 digits)
                - Video filename (will extract timestamp)
                - None (will prompt for input)
            fps: Frames per second of the video
        """
        self.fps = fps

        if video_start_time is None:
            print("Please provide the video start time:")
            print("Examples:")
            print("  - '2025-07-07T10:30:00.123'")
            print("  - '1735724709230' (milliseconds)")
            print("  - '1735724709' (seconds)")
            print("  - Video filename like '1735724709230.mp4'")
            user_input = input("Video start time: ").strip()
            video_start_time = user_input

        self.video_start_time_ms = self._parse_start_time(video_start_time)

        # Convert to datetime for display
        self.video_start_datetime = datetime.fromtimestamp(self.video_start_time_ms / 1000.0)
        print(f"📅 Video start time set to: {self.video_start_datetime.isoformat()}")
        print(f"📅 Video start timestamp: {self.video_start_time_ms} ms")

    def _parse_start_time(self, start_time):
        """Parse various start time formats into milliseconds since epoch"""

        if isinstance(start_time, datetime):
            return int(start_time.timestamp() * 1000)

        elif isinstance(start_time, (int, float)):
            # Check if it's already in milliseconds (13 digits) or seconds (10 digits)
            if start_time > 1e12:  # Likely milliseconds (13+ digits)
                return int(start_time)
            else:  # Likely seconds since epoch (10 digits)
                return int(start_time * 1000)

        elif isinstance(start_time, str):
            # First try to extract timestamp from filename
            timestamp_from_filename = self._extract_timestamp_from_filename(start_time)
            if timestamp_from_filename:
                return timestamp_from_filename

            # Try parsing as datetime string
            try:
                # Try parsing ISO format
                if 'T' in start_time:
                    dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                else:
                    # Try parsing common formats
                    for fmt in ['%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
                        try:
                            dt = datetime.strptime(start_time, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        # Try parsing as pure number string
                        try:
                            timestamp_num = int(start_time)
                            if timestamp_num > 1e12:  # Milliseconds
                                return timestamp_num
                            else:  # Seconds
                                return timestamp_num * 1000
                        except ValueError:
                            raise ValueError(f"Could not parse datetime string: {start_time}")

                return int(dt.timestamp() * 1000)

            except ValueError as e:
                raise ValueError(f"Invalid start time format: {start_time}. Error: {e}")

        else:
            raise ValueError(f"Unsupported start time type: {type(start_time)}")

    def _extract_timestamp_from_filename(self, filename_or_path):
        """Extract timestamp from video filename like '1735724709230.mp4'"""
        filename = os.path.basename(filename_or_path)
        name_without_ext = os.path.splitext(filename)[0]

        # Look for 13-digit timestamp (milliseconds) or 10-digit (seconds)
        timestamp_match = re.search(r'\b(\d{10,13})\b', name_without_ext)

        if timestamp_match:
            timestamp_str = timestamp_match.group(1)
            timestamp_num = int(timestamp_str)

            if len(timestamp_str) == 13:  # Milliseconds
                print(f"📅 Found millisecond timestamp in filename: {timestamp_num}")
                return timestamp_num
            elif len(timestamp_str) == 10:  # Seconds
                print(f"📅 Found second timestamp in filename: {timestamp_num}")
                return timestamp_num * 1000  # Convert to milliseconds

        return None

    def get_frame_timestamp_ms(self, frame_idx):
        """
        Get the actual timestamp when this frame was captured in the video (in milliseconds)

        Args:
            frame_idx: Frame index in the video (0-based)
        """
        # Calculate frame offset in milliseconds
        frame_offset_ms = int((frame_idx / self.fps) * 1000)

        # Add offset to video start time
        return self.video_start_time_ms + frame_offset_ms

    def get_formatted_timestamp(self, timestamp_ms):
        """Convert millisecond timestamp to readable format"""
        dt = datetime.fromtimestamp(timestamp_ms / 1000.0)
        return dt.isoformat()

    def get_frame_info(self, frame_idx):
        """Get comprehensive frame timing information"""
        frame_timestamp_ms = self.get_frame_timestamp_ms(frame_idx)

        return {
            'frame_idx': frame_idx,
            'frame_timestamp_ms': frame_timestamp_ms,
            'frame_timestamp_iso': self.get_formatted_timestamp(frame_timestamp_ms),
            'video_start_time_ms': self.video_start_time_ms,
            'video_start_time_iso': self.video_start_datetime.isoformat(),
            'frame_offset_seconds': frame_idx / self.fps,
            'frame_offset_ms': int((frame_idx / self.fps) * 1000),
            'fps': self.fps
        }


class PlantNetIdentifier:
    """Handles PlantNet API v2 interactions for plant identification."""

    def __init__(self, api_key, project="all"):
        """
        Args:
            api_key (str): Your PlantNet API key.
            project (str): One of: "all", "weurope", "canada", "australia"
        """
        self.api_key = api_key
        self.project = project
        self.base_url = f"https://my-api.plantnet.org/v2/identify/{project}"

        if not api_key or api_key == "YOUR_API_KEY":
            print("⚠️ WARNING: Please set your actual PlantNet API key!")
            print("Get your API key from: https://my.plantnet.org/")

    def crop_image_with_mask(self, image, mask, padding=20):
        """Crop a region from the image using the binary mask."""
        coords = np.column_stack(np.where(mask > 0))
        if len(coords) == 0:
            print("⚠️ No valid mask found.")
            return None

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        h, w = image.shape[:2]

        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)

        cropped = image[y_min:y_max, x_min:x_max]
        if cropped.size == 0:
            print("⚠️ Cropped image is empty.")
            return None
        return cropped

    def crop_image_with_bbox(self, image, bbox, padding=20):
        """Crop image using bounding box coordinates."""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            print("⚠️ Cropped image is empty.")
            return None
        return cropped

    def identify_plant(self, image_array, confidence_threshold=0.1):
        """Identify plant species in the given image using PlantNet API."""
        try:
            if isinstance(image_array, np.ndarray):
                if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                    pil_image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
                else:
                    pil_image = Image.fromarray(image_array[:, :, :3])
            else:
                pil_image = image_array

            if pil_image.size[0] < 50 or pil_image.size[1] < 50:
                print("⚠️ Image too small for identification (min size 50x50).")
                return None

            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='JPEG', quality=95)
            img_bytes = img_buffer.getvalue()

            url = f"{self.base_url}?api-key={self.api_key}"
            files = {'images': ('tree.jpg', img_bytes, 'image/jpeg')}
            data = {'organs': 'leaf'}  # You can also try 'flower', 'fruit', 'bark'

            print("🌿 Sending tree identification request to PlantNet...")
            response = requests.post(url, files=files, data=data, timeout=30)
            print(f"🌐 Response status: {response.status_code}")

            if response.status_code == 200:
                response_data = response.json()
                results = response_data.get('results', [])

                if not results:
                    return {
                        'species': 'No match found',
                        'common_names': [],
                        'confidence': 0,
                        'family': 'Unknown',
                        'genus': 'Unknown'
                    }

                filtered = [r for r in results if r.get('score', 0) >= confidence_threshold]
                best_match = filtered[0] if filtered else results[0]
                species_info = best_match.get('species', {})

                result = {
                    'species': species_info.get('scientificNameWithoutAuthor', 'Unknown'),
                    'common_names': species_info.get('commonNames', []),
                    'confidence': best_match.get('score', 0),
                    'family': species_info.get('family', {}).get('scientificNameWithoutAuthor', 'Unknown'),
                    'genus': species_info.get('genus', {}).get('scientificNameWithoutAuthor', 'Unknown')
                }

                if result['confidence'] < confidence_threshold:
                    result['species'] += ' (low confidence)'

                return result

            elif response.status_code == 401:
                print("❌ 401 Unauthorized - check your API key.")
            elif response.status_code == 429:
                print("❌ 429 Too Many Requests - rate limit exceeded. Waiting...")
                time.sleep(5)  # Wait longer before retry
                return {
                    'species': 'Rate Limit Exceeded',
                    'common_names': [],
                    'confidence': 0,
                    'family': 'Unknown',
                    'genus': 'Unknown'
                }
            else:
                print(f"❌ Unexpected API error ({response.status_code}): {response.text}")
                return {
                    'species': 'API Error',
                    'common_names': [],
                    'confidence': 0,
                    'family': 'Unknown',
                    'genus': 'Unknown'
                }

            return None

        except Exception as e:
            print(f"❌ Error during plant identification: {str(e)}")
            return {
                'species': 'Exception Error',
                'common_names': [],
                'confidence': 0,
                'family': 'Unknown',
                'genus': 'Unknown'
            }


class IntegratedTreeDetectionSystem:
    """Main class that integrates GroundedSAM2 with PlantNet API."""

    def __init__(self, plantnet_api_key, plantnet_project="all", video_start_time=None, video_fps=30.0):
        """Initialize the integrated system."""
        self.plant_identifier = PlantNetIdentifier(plantnet_api_key, plantnet_project)
        self.tree_results = {}  # Store all results

        # Initialize video timestamp generator
        self.video_timestamps = VideoTimestampGenerator(video_start_time, video_fps)

        self.setup_models()

    def setup_models(self):
        """Initialize SAM2 and GroundingDINO models."""
        # Environment settings
        torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Model initialization
        sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
        sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)
        self.image_predictor = SAM2ImagePredictor(sam2_image_model)

        # GroundingDINO model
        model_id = "IDEA-Research/grounding-dino-tiny"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

    def process_video_with_species_identification(self, video_dir, output_dir, text_prompt="tree.",
                                                step=20, confidence_threshold=0.1, video_start_time=None, video_fps=30.0):
        """
        Main pipeline: detect trees and identify species.

        Args:
            video_dir: Directory containing video frames
            output_dir: Output directory for results
            text_prompt: Text prompt for detection (default: "tree.")
            step: Frame sampling step
            confidence_threshold: Minimum confidence for PlantNet results
            video_start_time: When the video was originally recorded (optional if set in constructor)
            video_fps: Frames per second of the original video (optional if set in constructor)
        """

        # Update video timestamps if provided
        if video_start_time is not None:
            self.video_timestamps = VideoTimestampGenerator(video_start_time, video_fps)

        print("🚀 Starting integrated tree detection and species identification...")
        print(f"📹 Video timing: {self.video_timestamps.video_start_datetime.isoformat()} @ {self.video_timestamps.fps} fps")
        print(f"📹 Video start timestamp: {self.video_timestamps.video_start_time_ms} ms")

        # Setup directories
        CommonUtils.creat_dirs(output_dir)
        mask_data_dir = os.path.join(output_dir, "mask_data")
        json_data_dir = os.path.join(output_dir, "json_data")
        result_dir = os.path.join(output_dir, "result")
        species_data_dir = os.path.join(output_dir, "species_data")

        for dir_path in [mask_data_dir, json_data_dir, result_dir, species_data_dir]:
            CommonUtils.creat_dirs(dir_path)

        # Get frame names
        frame_names = [
            p for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        # Initialize video predictor
        inference_state = self.video_predictor.init_state(
            video_path=video_dir,
            offload_video_to_cpu=True,
            async_loading_frames=True
        )

        sam2_masks = MaskDictionaryModel()
        objects_count = 0

        print(f"📊 Total frames: {len(frame_names)}")

        # Process frames
        for start_frame_idx in range(0, len(frame_names), step):
            print(f"🔍 Processing frame {start_frame_idx}...")

            # Load and process image
            img_path = os.path.join(video_dir, frame_names[start_frame_idx])
            image = Image.open(img_path)
            image_base_name = frame_names[start_frame_idx].split(".")[0]

            # Run GroundingDINO detection
            detection_results = self._detect_trees(image, text_prompt)

            if detection_results is None:
                print(f"⚠️ No trees detected in frame {start_frame_idx}")
                continue

            # Get masks with SAM2
            masks, boxes, labels = self._get_masks(image, detection_results)

            if masks is None:
                print(f"⚠️ No masks generated for frame {start_frame_idx}")
                continue

            # Identify species for each detected tree
            species_results = self._identify_species_for_trees(
                np.array(image), masks, boxes, labels, start_frame_idx, confidence_threshold
            )

            # Process video tracking
            mask_dict = MaskDictionaryModel(
                promote_type="mask",
                mask_name=f"mask_{image_base_name}.npy"
            )
            mask_dict.add_new_frame_annotation(
                mask_list=torch.tensor(masks).to(self.device),
                box_list=torch.tensor(boxes),
                label_list=labels
            )

            objects_count = mask_dict.update_masks(
                tracking_annotation_dict=sam2_masks,
                iou_threshold=0.8,
                objects_count=objects_count
            )

            # Propagate tracking
            video_segments = self._propagate_tracking(
                inference_state, mask_dict, start_frame_idx, step, frame_names
            )

            # Save results with species information
            self._save_results_with_species(
                video_segments, mask_data_dir, json_data_dir, species_results, species_data_dir
            )

        # Generate final visualization and video
        self._create_final_output(video_dir, mask_data_dir, json_data_dir, result_dir, output_dir)

        # Save comprehensive results
        self._save_comprehensive_results(species_data_dir)

        print("✅ Processing complete!")

    def _detect_trees(self, image, text_prompt):
        """Detect trees using GroundingDINO."""
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]]
        )

        if len(results[0]["boxes"]) == 0:
            return None

        return results[0]

    def _get_masks(self, image, detection_results):
        """Generate masks using SAM2."""
        self.image_predictor.set_image(np.array(image.convert("RGB")))

        input_boxes = detection_results["boxes"]
        labels = detection_results["labels"]

        masks, scores, logits = self.image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # Convert mask shape to (n, H, W)
        if masks.ndim == 2:
            masks = masks[None]
        elif masks.ndim == 4:
            masks = masks.squeeze(1)

        return masks, input_boxes, labels

    def _identify_species_for_trees(self, image, masks, boxes, labels, frame_idx, confidence_threshold):
        """Identify species for each detected tree."""
        species_results = {}

        for i, (mask, box, label) in enumerate(zip(masks, boxes, labels)):
            print(f"🌲 Identifying species for tree {i+1} in frame {frame_idx}...")

            # Crop tree region using bounding box
            cropped_tree = self.plant_identifier.crop_image_with_bbox(image, box, padding=30)

            if cropped_tree is not None:
                # Add small delay to respect API rate limits
                time.sleep(0.5)

                # Identify species
                species_info = self.plant_identifier.identify_plant(
                    cropped_tree, confidence_threshold=confidence_threshold
                )

                tree_id = f"tree_{frame_idx}_{i}"

                # Get frame timing info based on video start time (in milliseconds)
                frame_info = self.video_timestamps.get_frame_info(frame_idx)

                species_results[tree_id] = {
                    **frame_info,  # Include all timing information
                    'tree_index': i,
                    'bounding_box': box.tolist(),
                    'detection_label': label,
                    'species_identification': species_info if species_info is not None else {
                        'species': 'Identification Failed',
                        'common_names': [],
                        'confidence': 0,
                        'family': 'Unknown',
                        'genus': 'Unknown'
                    }
                }

                if species_info:
                    print(f"✅ Identified: {species_info['species']} (confidence: {species_info['confidence']:.2f})")
                    print(f"🕐 Frame captured at: {frame_info['frame_timestamp_iso']} ({frame_info['frame_timestamp_ms']} ms)")
                else:
                    print("❌ Species identification failed")
            else:
                print("⚠️ Failed to crop tree region")
                tree_id = f"tree_{frame_idx}_{i}"

                # Get frame timing info even for failed crops
                frame_info = self.video_timestamps.get_frame_info(frame_idx)

                species_results[tree_id] = {
                    **frame_info,
                    'tree_index': i,
                    'bounding_box': box.tolist(),
                    'detection_label': label,
                    'species_identification': {
                        'species': 'Crop Failed',
                        'common_names': [],
                        'confidence': 0,
                        'family': 'Unknown',
                        'genus': 'Unknown'
                    }
                }

        return species_results

    def _propagate_tracking(self, inference_state, mask_dict, start_frame_idx, step, frame_names):
        """Propagate tracking across frames."""
        self.video_predictor.reset_state(inference_state)

        for object_id, object_info in mask_dict.labels.items():
            frame_idx, out_obj_ids, out_mask_logits = self.video_predictor.add_new_mask(
                inference_state,
                start_frame_idx,
                object_id,
                object_info.mask,
            )

        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(
            inference_state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx
        ):
            frame_masks = MaskDictionaryModel()

            for i, out_obj_id in enumerate(out_obj_ids):
                out_mask = (out_mask_logits[i] > 0.0)
                object_info = ObjectInfo(
                    instance_id=out_obj_id,
                    mask=out_mask[0],
                    class_name=mask_dict.get_target_class_name(out_obj_id)
                )
                object_info.update_box()
                frame_masks.labels[out_obj_id] = object_info

                image_base_name = frame_names[out_frame_idx].split(".")[0]
                frame_masks.mask_name = f"mask_{image_base_name}.npy"
                frame_masks.mask_height = out_mask.shape[-2]
                frame_masks.mask_width = out_mask.shape[-1]

            video_segments[out_frame_idx] = frame_masks

        return video_segments

    def _save_results_with_species(self, video_segments, mask_data_dir, json_data_dir, species_results, species_data_dir):
        """Save tracking results along with species information."""
        for frame_idx, frame_masks_info in video_segments.items():
            # Save mask data
            mask = frame_masks_info.labels
            mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)

            for obj_id, obj_info in mask.items():
                mask_img[obj_info.mask == True] = obj_id

            mask_img = mask_img.numpy().astype(np.uint16)
            np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)

            # Save JSON data with species information
            json_data = copy.deepcopy(frame_masks_info.to_dict())

            # Add species information if available
            frame_species_info = {}
            for tree_id, species_data in species_results.items():
                if species_data['frame_idx'] == frame_idx:
                    frame_species_info[tree_id] = species_data

            json_data['species_identifications'] = frame_species_info

            # Add video-based timing information (in milliseconds)
            frame_info = self.video_timestamps.get_frame_info(frame_idx)
            json_data.update(frame_info)

            print(f"[✔] Frame {frame_idx} timestamped as: {frame_info['frame_timestamp_ms']} ms ({frame_info['frame_timestamp_iso']})")

            json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
            with open(json_data_path, "w") as f:
                json.dump(json_data, f, indent=2)

        # Save species results separately
        if species_results:
            species_file = os.path.join(species_data_dir, f"species_frame_{min(species_results.values(), key=lambda x: x['frame_idx'])['frame_idx']}.json")
            with open(species_file, "w") as f:
                json.dump(species_results, f, indent=2)

            # Update global results
            self.tree_results.update(species_results)

    def _create_final_output(self, video_dir, mask_data_dir, json_data_dir, result_dir, output_dir):
        """Create final visualization and video."""
        # Draw results
        CommonUtils.draw_masks_and_box_with_supervision(video_dir, mask_data_dir, json_data_dir, result_dir)

        # Create output video
        output_video_path = os.path.join(output_dir, "tree_detection_with_species.mp4")
        create_video_from_images(result_dir, output_video_path, frame_rate=30)
        print(f"📹 Output video saved: {output_video_path}")

    def _save_comprehensive_results(self, species_data_dir):
        """Save comprehensive results summary."""
        summary_file = os.path.join(species_data_dir, "comprehensive_tree_analysis.json")

        # Create summary statistics
        species_count = {}
        total_trees = len(self.tree_results)

        for tree_id, tree_data in self.tree_results.items():
            species_info = tree_data.get('species_identification', {})

            # Handle None species_info
            if species_info is None:
                species_name = 'Identification Failed'
            else:
                species_name = species_info.get('species', 'Unknown')

            if species_name not in species_count:
                species_count[species_name] = 0
            species_count[species_name] += 1

        summary = {
            'analysis_summary': {
                'total_trees_detected': total_trees,
                'unique_species_found': len(species_count),
                'species_distribution': species_count,
                'analysis_timestamp_ms': int(time.time() * 1000),
                'analysis_timestamp_iso': datetime.now().isoformat(),
                'video_start_time_ms': self.video_timestamps.video_start_time_ms,
                'video_start_time_iso': self.video_timestamps.video_start_datetime.isoformat(),
                'video_fps': self.video_timestamps.fps
            },
            'detailed_results': self.tree_results
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"📊 Comprehensive results saved: {summary_file}")
        print(f"🌳 Total trees detected: {total_trees}")
        print(f"🌿 Unique species found: {len(species_count)}")
        for species, count in species_count.items():
            print(f"   - {species}: {count} trees")


# Usage example
if __name__ == "__main__":
    # Configuration
    PLANTNET_API_KEY = "Your_API_Key"
    PLANTNET_PROJECT = "all"  # or "weurope", "canada", "australia"

    # Video timing information - USE YOUR ACTUAL VIDEO FILENAME TIMESTAMP
    VIDEO_START_TIME_MS = 1751270034249 # Your video filename timestamp in milliseconds
    VIDEO_FPS = 30.0  # Frames per second of your video

    # Alternative ways to specify video start time:
    # VIDEO_START_TIME_MS = "1735724709230"  # As string
    # VIDEO_START_TIME_MS = "1735724709230.mp4"  # As filename (will extract timestamp)
    # VIDEO_START_TIME_MS = datetime(2025, 1, 1, 14, 45, 9, 230000)  # datetime object
    # VIDEO_START_TIME_MS = "2025-01-01T14:45:09.230"  # ISO format
    # VIDEO_START_TIME_MS = None  # Will prompt for input

    # Initialize the integrated system with video timing
    tree_system = IntegratedTreeDetectionSystem(
        PLANTNET_API_KEY,
        PLANTNET_PROJECT,
        video_start_time=VIDEO_START_TIME_MS,
        video_fps=VIDEO_FPS
    )

    # Process video with tree detection and species identification
    video_dir = "frames/"
    output_dir = "./outputs_with_species"

    tree_system.process_video_with_species_identification(
        video_dir=video_dir,
        output_dir=output_dir,
        text_prompt="tree.",
        step=100,
        confidence_threshold=0.1
        # video_start_time and video_fps are optional here if already set in constructor
    )