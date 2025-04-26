import torch
import torchvision
from torchvision import transforms
import cv2
import numpy as np
from pathlib import Path
import logging
import urllib.request
from torchvision import models
from torch import nn

logger = logging.getLogger(__name__)

class SceneUnderstanding:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize scene understanding models"""
        # Initialize scene classification model (ResNet50 pre-trained on Places365)
        self.scene_classifier = models.resnet50(pretrained=False)
        num_ftrs = self.scene_classifier.fc.in_features
        self.scene_classifier.fc = nn.Linear(num_ftrs, 365)  # Places365 has 365 categories
        
        # Load Places365 weights
        model_url = 'http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar'
        state_dict = torch.hub.load_state_dict_from_url(model_url, progress=True, map_location=self.device)
        
        # Remove 'module.' prefix from state dict keys
        new_state_dict = {}
        for k, v in state_dict['state_dict'].items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        self.scene_classifier.load_state_dict(new_state_dict)
        self.scene_classifier.eval()
        self.scene_classifier.to(self.device)
        
        # Load Places365 categories
        self.categories = self._load_categories()
        
        # Scene segmentation model (temporarily disabled)
        self.segmenter = None
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def _load_categories(self):
        """Load Places365 categories"""
        # Complete set of Places365 categories
        categories = {
            0: "airfield",
            1: "airplane_cabin",
            2: "airport_terminal",
            3: "alcove",
            4: "alley",
            5: "amphitheater",
            6: "amusement_arcade",
            7: "amusement_park",
            8: "apartment_building/outdoor",
            9: "aquarium",
            10: "aqueduct",
            11: "arcade",
            12: "arch",
            13: "archaelogical_excavation",
            14: "archive",
            15: "arena/hockey",
            16: "arena/performance",
            17: "arena/rodeo",
            18: "army_base",
            19: "art_gallery",
            20: "art_school",
            21: "art_studio",
            22: "artists_loft",
            23: "assembly_line",
            24: "athletic_field/outdoor",
            25: "atrium/public",
            26: "attic",
            27: "auditorium",
            28: "auto_factory",
            29: "auto_showroom",
            30: "badlands",
            31: "bakery/shop",
            32: "balcony/exterior",
            33: "balcony/interior",
            34: "ball_pit",
            35: "ballroom",
            36: "bamboo_forest",
            37: "bank_vault",
            38: "banquet_hall",
            39: "bar",
            40: "barn",
            41: "barndoor",
            42: "baseball_field",
            43: "basement",
            44: "basketball_court/indoor",
            45: "bathroom",
            46: "bazaar/indoor",
            47: "bazaar/outdoor",
            48: "beach",
            49: "beach_house",
            50: "beauty_salon",
            51: "bedroom",
            52: "beer_garden",
            53: "beer_hall",
            54: "berth",
            55: "biology_laboratory",
            56: "boardwalk",
            57: "boat_deck",
            58: "boathouse",
            59: "bookstore",
            60: "booth/indoor",
            61: "botanical_garden",
            62: "bow_window/indoor",
            63: "bowling_alley",
            64: "boxing_ring",
            65: "bridge",
            66: "building_facade",
            67: "bullring",
            68: "burial_chamber",
            69: "bus_interior",
            70: "bus_station/indoor",
            71: "butchers_shop",
            72: "butte",
            73: "cabin/outdoor",
            74: "cafeteria",
            75: "campsite",
            76: "campus",
            77: "canal/natural",
            78: "canal/urban",
            79: "candy_store",
            80: "canyon",
            81: "car_interior",
            82: "carrousel",
            83: "castle",
            84: "catacomb",
            85: "cemetery",
            86: "chalet",
            87: "chemistry_lab",
            88: "childs_room",
            89: "church/indoor",
            90: "church/outdoor",
            91: "classroom",
            92: "clean_room",
            93: "cliff",
            94: "closet",
            95: "clothing_store",
            96: "coast",
            97: "cockpit",
            98: "coffee_shop",
            99: "computer_room",
            100: "conference_center",
            101: "conference_room",
            102: "construction_site",
            103: "corn_field",
            104: "corral",
            105: "corridor",
            106: "cottage",
            107: "courthouse",
            108: "courtyard",
            109: "creek",
            110: "crevasse",
            111: "crosswalk",
            112: "dam",
            113: "delicatessen",
            114: "department_store",
            115: "desert/sand",
            116: "desert/vegetation",
            117: "desert_road",
            118: "diner/outdoor",
            119: "dining_hall",
            120: "dining_room",
            121: "discotheque",
            122: "doorway/outdoor",
            123: "dorm_room",
            124: "downtown",
            125: "dressing_room",
            126: "driveway",
            127: "drugstore",
            128: "elevator/door",
            129: "elevator_lobby",
            130: "elevator_shaft",
            131: "embassy",
            132: "engine_room",
            133: "entrance_hall",
            134: "escalator/indoor",
            135: "excavation",
            136: "fabric_store",
            137: "farm",
            138: "fastfood_restaurant",
            139: "field/cultivated",
            140: "field/wild",
            141: "field_road",
            142: "fire_escape",
            143: "fire_station",
            144: "fishpond",
            145: "flea_market/indoor",
            146: "florist_shop/indoor",
            147: "food_court",
            148: "football_field",
            149: "forest/broadleaf",
            150: "forest_path",
            151: "forest_road",
            152: "formal_garden",
            153: "fountain",
            154: "galley",
            155: "garage/indoor",
            156: "garage/outdoor",
            157: "gas_station",
            158: "gazebo/exterior",
            159: "general_store/indoor",
            160: "general_store/outdoor",
            161: "gift_shop",
            162: "glacier",
            163: "golf_course",
            164: "greenhouse/indoor",
            165: "greenhouse/outdoor",
            166: "grotto",
            167: "gymnasium/indoor",
            168: "hangar/indoor",
            169: "hangar/outdoor",
            170: "harbor",
            171: "hardware_store",
            172: "hayfield",
            173: "heliport",
            174: "highway",
            175: "home_office",
            176: "home_theater",
            177: "hospital",
            178: "hospital_room",
            179: "hot_spring",
            180: "hotel/outdoor",
            181: "hotel_room",
            182: "house",
            183: "hunting_lodge/outdoor",
            184: "ice_cream_parlor",
            185: "ice_floe",
            186: "ice_shelf",
            187: "ice_skating_rink/indoor",
            188: "ice_skating_rink/outdoor",
            189: "iceberg",
            190: "igloo",
            191: "industrial_area",
            192: "inn/outdoor",
            193: "islet",
            194: "jacuzzi/indoor",
            195: "jail_cell",
            196: "japanese_garden",
            197: "jewelry_shop",
            198: "mountain_path",
            199: "mountain_snowy",
            200: "movie_theater/indoor",
            201: "museum/indoor",
            202: "museum/outdoor",
            203: "music_studio",
            204: "natural_history_museum",
            205: "nursery",
            206: "nursing_home",
            207: "oast_house",
            208: "ocean",
            209: "office",
            210: "office_building",
            211: "office_cubicles",
            212: "oilrig",
            213: "operating_room",
            214: "orchard",
            215: "orchestra_pit",
            216: "pagoda",
            217: "palace",
            218: "pantry",
            219: "park",
            220: "parking_garage/indoor",
            221: "parking_garage/outdoor",
            222: "parking_lot",
            223: "pasture",
            224: "patio",
            225: "pavilion",
            226: "pet_shop",
            227: "pharmacy",
            228: "phone_booth",
            229: "physics_laboratory",
            230: "picnic_area",
            231: "pier",
            232: "pizzeria",
            233: "playground",
            234: "playroom",
            235: "plaza",
            236: "pond",
            237: "porch",
            238: "promenade",
            239: "pub/indoor",
            240: "racecourse",
            241: "raceway",
            242: "raft",
            243: "railroad_track",
            244: "rainforest",
            245: "reception",
            246: "recreation_room",
            247: "repair_shop",
            248: "residential_neighborhood",
            249: "restaurant",
            250: "restaurant_kitchen",
            251: "restaurant_patio",
            252: "rice_paddy",
            253: "river",
            254: "rock_arch",
            255: "roof_garden",
            256: "rope_bridge",
            257: "ruin",
            258: "runway",
            259: "sandbox",
            260: "sauna",
            261: "schoolhouse",
            262: "science_museum",
            263: "server_room",
            264: "shed",
            265: "shoe_shop",
            266: "shopfront",
            267: "shopping_mall/indoor",
            268: "shower",
            269: "ski_resort",
            270: "ski_slope",
            271: "sky",
            272: "skyscraper",
            273: "slum",
            274: "snowfield",
            275: "soccer_field",
            276: "stable",
            277: "stadium/baseball",
            278: "stadium/football",
            279: "stadium/soccer",
            280: "stage/indoor",
            281: "stage/outdoor",
            282: "staircase",
            283: "storage_room",
            284: "street",
            285: "subway_station/platform",
            286: "supermarket",
            287: "sushi_bar",
            288: "swamp",
            289: "swimming_hole",
            290: "swimming_pool/indoor",
            291: "swimming_pool/outdoor",
            292: "synagogue/outdoor",
            293: "television_room",
            294: "television_studio",
            295: "temple/asia",
            296: "throne_room",
            297: "ticket_booth",
            298: "topiary_garden",
            299: "tower",
            300: "toyshop",
            301: "train_interior",
            302: "train_station/platform",
            303: "tree_farm",
            304: "tree_house",
            305: "trench",
            306: "tundra",
            307: "underwater/ocean_deep",
            308: "utility_room",
            309: "valley",
            310: "vegetable_garden",
            311: "veterinarians_office",
            312: "viaduct",
            313: "village",
            314: "vineyard",
            315: "volcano",
            316: "volleyball_court/outdoor",
            317: "waiting_room",
            318: "warehouse/indoor",
            319: "warehouse/outdoor",
            320: "wasteland",
            321: "water_park",
            322: "water_tower",
            323: "waterfall",
            324: "watering_hole",
            325: "wave",
            326: "wet_bar",
            327: "wheat_field",
            328: "wind_farm",
            329: "windmill",
            330: "yard",
            331: "youth_hostel",
            332: "zen_garden",
            333: "apartment_building/indoor",
            334: "artists_studio",
            335: "auditorium/indoor",
            336: "ballroom/dance_studio",
            337: "basement/indoor",
            338: "bathroom/indoor",
            339: "bedroom/indoor",
            340: "biology_laboratory",
            341: "bookstore/indoor",
            342: "building_facade",
            343: "campus/outdoor",
            344: "church/indoor",
            345: "classroom/indoor",
            346: "clean_room",
            347: "conference_center",
            348: "corridor/indoor",
            349: "courthouse/indoor",
            350: "dining_room/indoor",
            351: "elevator/interior",
            352: "factory/indoor",
            353: "food_court",
            354: "game_room/indoor",
            355: "garage/indoor",
            356: "greenhouse/indoor",
            357: "gymnasium/indoor",
            358: "home_office/indoor",
            359: "hospital_room/indoor",
            360: "hotel_room/indoor",
            361: "kitchen/indoor",
            362: "laboratory/indoor",
            363: "library/indoor",
            364: "living_room/indoor"
        }
        return categories
    
    def preprocess_frame(self, frame):
        """Preprocess a video frame for analysis"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Apply transformations
        frame_tensor = self.transform(frame_rgb)
        return frame_tensor.unsqueeze(0).to(self.device)
    
    def classify_scene(self, frame):
        """Classify the scene type of a frame"""
        try:
            # Preprocess frame
            frame_tensor = self.preprocess_frame(frame)
            
            # Get scene classification
            with torch.no_grad():
                outputs = self.scene_classifier(frame_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                top_prob, top_catid = torch.topk(probabilities, 3)
            
            # Map categories to names
            top_categories = []
            for prob, catid in zip(top_prob, top_catid):
                catid = catid.item()
                category_name = self.categories.get(catid, f"unknown_{catid}")
                top_categories.append({
                    'category': category_name,
                    'probability': prob.item()
                })
            
            return {
                'top_categories': top_categories,
                'raw_output': outputs.cpu().numpy()
            }
            
        except Exception as e:
            logger.error(f"Error in scene classification: {str(e)}")
            return None
    
    def segment_scene(self, frame):
        """Segment the scene into semantic regions"""
        # Temporarily return None until segmentation model is implemented
        return None
    
    def analyze_frame(self, frame):
        """Complete scene analysis of a frame"""
        scene_class = self.classify_scene(frame)
        scene_seg = self.segment_scene(frame)
        
        return {
            'classification': scene_class,
            'segmentation': scene_seg
        }

if __name__ == "__main__":
    # Test the scene understanding module
    analyzer = SceneUnderstanding()
    
    # Test with a sample frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Placeholder frame
    results = analyzer.analyze_frame(test_frame)
    
    print("Scene understanding test results:")
    print(results) 