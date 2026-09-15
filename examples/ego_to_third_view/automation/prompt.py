"""Construct one continuous scene-specific prompt from reviewed structured facts."""

def build_prompt(scene,events,duration):
    descriptions={o['id']:o.get('description',o['id']) for o in scene['objects']}
    actions=[]
    for event in events:
        actions.append(f'At {event["time_s"]:.3f}s: {event.get("action",event["id"])} '
                       f'{descriptions.get(event.get("object_id"), "")}.')
    return (
        f'integrated_multimodal_description: One uninterrupted {duration:.3f}-second photoreal recording. '
        'Reference video 1 is authoritative for fixed third-person projection, source-derived operator location, '
        'hand/object paths, contact phases and precise timing. Reference image 1 supplies appearance only. '
        'The actor remains at the source ego operator location shown in the control, not at the observer camera. '
        'Both arms attach to one shared stable torso with gentle source-consistent yaw, no hand-driven body translation, '
        'bobbing or swaying. Human upper-arm and forearm lengths remain constant. No limb stretching, elbow flips, '
        'disconnected arms, duplicate hands or unnatural wrist twists. '
        'No camera pan, zoom, scene cut, reset or reframe. Preserve object count, identity, dimensions, orientation, '
        'which surfaces are visible, occlusion, and supported resting positions. Contacts move together without slipping, '
        'floating or penetration; do not invent a final release if the control ends while holding. '
        'Suppress high-frequency tremor without shifting action times. Preserve the supplied scene, materials and clothing. '
        +str(scene.get('appearance_description',''))+' '
        +' '.join(actions)+' overall_soundscape: quiet environmental ambience. non_diegetic_music: none.'
    )
