mesh_annotations_derived_flip_map = {
        # Midline points (verified or assumed from MESH_ANNOTATIONS)
        168: 168, # midwayBetweenEyes
        1: 1,     # noseTip
        2: 2,     # noseBottom
        0: 0,     # lipsUpperOuter (often midline)
        13: 13,   # lipsUpperInner (often midline)
        17: 17,   # lipsLowerOuter (often midline)
        14: 14,   # lipsLowerInner (often midline)
        10: 10,   # silhouette (top of head - assumed midline)
        152: 152, # silhouette (bottom of chin - assumed midline)

        # Lips (excluding assumed midline points above)
        # lipsUpperOuter: [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        61: 291, 291: 61,
        185: 409, 409: 185,
        40: 270, 270: 40,
        39: 269, 269: 39,
        37: 267, 267: 37,
        # lipsLowerOuter: [146, 91, 181, 84, 17, 314, 405, 321, 375, 291] (291 is also in upper)
        146: 375, 375: 146,
        91: 321, 321: 91,
        181: 405, 405: 181,
        84: 314, 314: 84,
        # lipsUpperInner: [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
        78: 308, 308: 78, # Note: 78 and 308 are often mouth corners, shared by upper/lower inner/outer in some schemes
        191: 415, 415: 191,
        80: 310, 310: 80,
        81: 311, 311: 81,
        82: 312, 312: 82,
        # lipsLowerInner: [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
        95: 324, 324: 95,
        88: 318, 318: 88,
        178: 402, 402: 178,
        87: 317, 317: 87,

        # Eyes
        246: 466, 466: 246, 161: 388, 388: 161, 160: 387, 387: 160, 159: 386, 386: 159, 158: 385, 385: 158, 157: 384, 384: 157, 173: 398, 398: 173, # rightEyeUpper0 vs leftEyeUpper0
        33: 263, 263: 33, 7: 249, 249: 7, 163: 390, 390: 163, 144: 373, 373: 144, 145: 374, 374: 145, 153: 380, 380: 153, 154: 381, 381: 154, 155: 382, 382: 155, 133: 362, 362: 133, # rightEyeLower0 vs leftEyeLower0
        247: 467, 467: 247, 30: 260, 260: 30, 29: 259, 259: 29, 27: 257, 257: 27, 28: 258, 258: 28, 56: 286, 286: 56, 190: 414, 414: 190, # rightEyeUpper1 vs leftEyeUpper1
        130: 359, 359: 130, 25: 255, 255: 25, 110: 339, 339: 110, 24: 254, 254: 24, 23: 253, 253: 23, 22: 252, 252: 22, 26: 256, 256: 26, 112: 341, 341: 112, 243: 463, 463: 243, # rightEyeLower1 vs leftEyeLower1
        113: 342, 342: 113, 225: 445, 445: 225, 224: 444, 444: 224, 223: 443, 443: 223, 222: 442, 442: 222, 221: 441, 441: 221, 189: 413, 413: 189, # rightEyeUpper2 vs leftEyeUpper2
        226: 446, 446: 226, 31: 261, 261: 31, 228: 448, 448: 228, 229: 449, 449: 229, 230: 450, 450: 230, 231: 451, 451: 231, 232: 452, 452: 232, 233: 453, 453: 233, 244: 464, 464: 244, # rightEyeLower2 vs leftEyeLower2
        143: 372, 372: 143, 111: 340, 340: 111, 117: 346, 346: 117, 118: 347, 347: 118, 119: 348, 348: 119, 120: 349, 349: 120, 121: 350, 350: 121, 128: 357, 357: 128, 245: 465, 465: 245, # rightEyeLower3 vs leftEyeLower3

        # Eyebrows
        156: 383, 383: 156, 70: 300, 300: 70, 63: 293, 293: 63, 105: 334, 334: 105, 66: 296, 296: 66, 107: 336, 336: 107, 55: 285, 285: 55, 193: 417, 417: 193, # rightEyebrowUpper vs leftEyebrowUpper
        35: 265, 265: 35, 124: 353, 353: 124, 46: 276, 276: 46, 53: 283, 283: 53, 52: 282, 282: 52, 65: 295, 295: 65, # rightEyebrowLower vs leftEyebrowLower

        # Iris
        473: 468, 468: 473, 474: 469, 469: 474, 475: 470, 470: 475, 476: 471, 471: 476, 477: 472, 472: 477, # rightEyeIris vs leftEyeIris

        # Nose Corners
        98: 327, 327: 98, # noseRightCorner vs noseLeftCorner

        # Cheeks
        205: 425, 425: 205, # rightCheek vs leftCheek

        # USER ACTION REQUIRED for 'silhouette':
        # The 'silhouette' array:
        # [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        #  397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        #  172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        # Example (needs verification by user by visualizing landmarks):
        # 338: 109, 109: 338, # (Assuming these are a pair)
        # 297: 67, 67: 297,   # (Assuming these are a pair)
        # ... and so on for all silhouette points. Some might be midline.
        # Points 10 and 152 are already added as midline.
    }