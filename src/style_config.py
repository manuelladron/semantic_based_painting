style_parameters = {
    'realistic': {
        'budgets': [9, 49, 64, 81],
        'number_natural_patches': [40, 60, 60],
        'patch_strategy_detail': 'uniform',
        'use_segmentation_mask': False,
        'filter_strokes': False,
        'return_segmented_areas': False
    },
    'abstract': {
        'budgets': [9, 16, 16, 9],
        'number_natural_patches': [25, 25, 50],
        'patch_strategy_detail': 'natural',
        'use_segmentation_mask': True,
        'filter_strokes': True,
        'return_segmented_areas': True
    },
    'expressionist': {
        'iter_steps': [500, 500],
        'brush_sizes': [0.8, 0.8],
        'budgets': [9, 9],
        'number_natural_patches': [9],
        'patch_strategy_detail': 'natural',
        'use_segmentation_mask': False,
        'filter_strokes': False,
        'return_segmented_areas': False,
        'start_using_masks' : -1,
        'start_natural_level' : 1,
    },
    'painterly': {
        'budgets': [9, 16, 16, 9],
        'number_natural_patches': [25, 25, 25],
        'patch_strategy_detail': 'natural',
        'use_segmentation_mask': True,
        'filter_strokes': True,
        'return_segmented_areas': True
    },
}
