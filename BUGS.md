● Here's a comprehensive comparison of all 18 C modules against their Python translations. I'll organize findings by severity.

  ---
  CRITICAL BUGS (Will crash or produce wrong results)

  Constants & Data Structures

  ┌────────────────────────┬──────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────┐
  │          Bug           │         Location         │                                               Issue                                               │
  ├────────────────────────┼──────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ POSI = 4 instead of 80 │ constants.py:3           │ C defines POSI 80 for decision arrays. Python uses 4, truncating candidate lists                  │
  ├────────────────────────┼──────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Missing sentinel       │ constants.py             │ PT_UNUSED (-999), CORRES_NONE (-1), PREV_NONE (-1), NEXT_NONE (-2), TR_UNUSED (-1), MAX_TARGETS,  │
  │ constants              │                          │ COORD_UNUSED all missing                                                                          │
  ├────────────────────────┼──────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ MmLut.rw is float      │ calibration.py:133       │ C declares int rw, Python uses float rw = 2.0                                                     │
  ├────────────────────────┼──────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Pathinfo field names   │ tracking_frame_buf.py:70 │ Defines prev_frame/next_frame but track3d.py uses .prev/.next — will crash with AttributeError    │
  ├────────────────────────┼──────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Corres.p fixed at 4    │ tracking_frame_buf.py:62 │ C always has p[4], Python allows variable num_cams — breaks 4-camera assumptions                  │
  └────────────────────────┴──────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────┘

  Core Geometry (corrupts all downstream)

  ┌───────────────────────────────────┬───────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────┐
  │                Bug                │     Location      │                                             Issue                                             │
  ├───────────────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┤
  │ sr -= iz instead of sr -= ir      │ multimed.py:296   │ Bilinear interpolation in get_mmf_from_mmlut uses wrong index variable — corrupts all         │
  │                                   │                   │ LUT-based multimedia corrections                                                              │
  ├───────────────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Missing ext_z0 parameter          │ multimed.py:18    │ multimed_nlay() can't call iterative version because ext_z0 isn't passed through              │
  ├───────────────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Wrong camera center in            │ imgcoord.py:54-56 │ Passes original ext_x0, ext_y0 instead of the transformed camera center from trans_cam_point  │
  │ flat_image_coord                  │                   │                                                                                               │
  ├───────────────────────────────────┼───────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┤
  │ init_mmlut is a stub              │ multimed.py:418   │ Raises NotImplementedError — Python can't build multimedia LUTs                               │
  └───────────────────────────────────┴───────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────┘

  Image Processing

  ┌──────────────────────────┬─────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────┐
  │           Bug            │          Location           │                                            Issue                                             │
  ├──────────────────────────┼─────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
  │ filter_3 boundary        │ image_processing.py:52-68   │ Processes ALL pixels including borders; C skips first and last rows entirely                 │
  │ handling                 │                             │                                                                                              │
  ├──────────────────────────┼─────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
  │ lowpass_3 same boundary  │ image_processing.py:87-101  │ Same issue as filter_3                                                                       │
  │ bug                      │                             │                                                                                              │
  ├──────────────────────────┼─────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────┤
  │ fast_box_blur multiple   │ image_processing.py:131-156 │ Wrong edge weighting, adds 1 line/iteration instead of 2 in column accumulation, wrong       │
  │ bugs                     │                             │ formula for last lines                                                                       │
  └──────────────────────────┴─────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────┘

  Correspondences (fundamentally broken)

  ┌───────────────────────────────────┬────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────┐
  │                Bug                │          Location          │                                        Issue                                         │
  ├───────────────────────────────────┼────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ Data structure mismatch           │ correspondences.py         │ C uses list[cam1][cam2][target_idx] fixed arrays; Python uses flat lists requiring   │
  │                                   │                            │ lookup — breaks all matching                                                         │
  ├───────────────────────────────────┼────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ Missing distance-weighted scoring │ correspondences.py:249-257 │ C computes sum(corr)/sum(dist); Python uses mean(corr) — fundamentally different     │
  │                                   │                            │ scoring metric                                                                       │
  ├───────────────────────────────────┼────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ Extra bidirectional check in pair │ correspondences.py:371-388 │ Python requires cam2→cam1 also unambiguous; C only checks cam1→cam2 — stricter,      │
  │  matching                         │                            │ finds fewer pairs                                                                    │
  ├───────────────────────────────────┼────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────┤
  │ Missing tnr write-back            │ correspondences.py         │ C writes correspondence index back into target.tnr; Python doesn't                   │
  └───────────────────────────────────┴────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────┘

  Tracking (mostly stubs)

  ┌────────────────────────────────────────────────┬─────────────────┬──────────────────────────────────────────────────────────────────────┐
  │                      Bug                       │    Location     │                                Issue                                 │
  ├────────────────────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ candsearch_in_pix returns array, not count     │ track.py:70-108 │ C returns candidate count; Python returns index array — API mismatch │
  ├────────────────────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ Wrong tnr sentinel in candsearch_in_pix        │ track.py:88     │ Checks tnr != -999 but C uses TR_UNUSED = -1                         │
  ├────────────────────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ trackcorr_c_loop is a stub                     │ track.py        │ Entire tracking algorithm is unimplemented                           │
  ├────────────────────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ trackback_c is a stub                          │ track.py        │ Backward tracking unimplemented                                      │
  ├────────────────────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ Missing fb.next(), fb.write_frame_from_start() │ track3d.py:138  │ Called but never defined in tracking_frame_buf.py                    │
  ├────────────────────────────────────────────────┼─────────────────┼──────────────────────────────────────────────────────────────────────┤                 
  │ Missing volumedimension() call                 │ tracking_run.py │ ymin/ymax never computed                                             │
  └────────────────────────────────────────────────┴─────────────────┴──────────────────────────────────────────────────────────────────────┘                 
                  
  Orientation (skeleton only)                                                                                                                                 
                  
  ┌───────────────────────────────────────────┬────────────────────────┬──────────────────────────────────────────────────────────────────────────────────┐
  │                    Bug                    │        Location        │                                      Issue                                       │
  ├───────────────────────────────────────────┼────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
  │ skew_midpoint is a stub                   │ orientation.py:107     │ Returns simple vertex average, ignoring ray directions entirely                  │
  ├───────────────────────────────────────────┼────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
  │ orient missing ~90% of logic              │ orientation.py:257-395 │ No interior params, no distortion derivatives, no glass interface, no weight     │   
  │                                           │                        │ matrix                                                                           │   
  ├───────────────────────────────────────────┼────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤   
  │ raw_orient incomplete                     │ orientation.py:398-473 │ Missing Gauss-Markov solution                                                    │   
  ├───────────────────────────────────────────┼────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
  │ num_deriv_exterior uses central           │ orientation.py:171-254 │ C uses forward differences — gives different numerical results                   │   
  │ differences                               │                        │                                                                                  │
  └───────────────────────────────────────────┴────────────────────────┴──────────────────────────────────────────────────────────────────────────────────┘   
                  
  ---
  MODERATE ISSUES

  ┌───────────────────────────────────────────────────┬─────────────────────────┬──────────────────────────────────────────────────────────────────────┐
  │                        Bug                        │        Location         │                                Issue                                 │
  ├───────────────────────────────────────────────────┼─────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ angle_acc missing 180-degree case                 │ track.py:60-66          │ C returns 200 gon for opposite directions; Python doesn't check this │
  ├───────────────────────────────────────────────────┼─────────────────────────┼──────────────────────────────────────────────────────────────────────┤
  │ candsearch_in_pix missing binary search           │ track.py                │ C uses binary search by y-coordinate; Python iterates linearly       │      
  ├───────────────────────────────────────────────────┼─────────────────────────┼──────────────────────────────────────────────────────────────────────┤      
  │ Quicksort replaced with stable sort               │ correspondences.py:414  │ Different ordering for equal-correlation values                      │      
  ├───────────────────────────────────────────────────┼─────────────────────────┼──────────────────────────────────────────────────────────────────────┤      
  │ copy_images signature mismatch                    │ image_processing.py:272 │ C copies one image; Python copies a list                             │
  ├───────────────────────────────────────────────────┼─────────────────────────┼──────────────────────────────────────────────────────────────────────┤      
  │ sortgrid returns None instead of Target(pnr=-999) │ sortgrid.py:111-114     │ Will crash when accessing .pnr on unmatched entries                  │
  ├───────────────────────────────────────────────────┼─────────────────────────┼──────────────────────────────────────────────────────────────────────┤      
  │ C bug: compare_sequence_par                       │ parameters.c:102        │ Compares sp1 to itself instead of sp2 (C bug, not Python)            │
  └───────────────────────────────────────────────────┴─────────────────────────┴──────────────────────────────────────────────────────────────────────┘      
                  
  ---                                                                                                                                                         
  MODULE STATUS SUMMARY
                       
  ┌────────────────────┬──────────────────────┬────────────────────────────────────────────────────────────┐
  │       Module       │        Status        │                         Key Issue                          │                                                  
  ├────────────────────┼──────────────────────┼────────────────────────────────────────────────────────────┤                                                  
  │ vec_utils          │ Good                 │ Complete match                                             │                                                  
  ├────────────────────┼──────────────────────┼────────────────────────────────────────────────────────────┤                                                  
  │ lsqadj             │ Good                 │ Minor: norm_cross declared in C header but not in C source │                                                  
  ├────────────────────┼──────────────────────┼────────────────────────────────────────────────────────────┤                                                  
  │ calibration        │ Good                 │ Minor: rw type mismatch                                    │                                                  
  ├────────────────────┼──────────────────────┼────────────────────────────────────────────────────────────┤                                                  
  │ parameters         │ Good                 │ Field order differs but named access works                 │
  ├────────────────────┼──────────────────────┼────────────────────────────────────────────────────────────┤                                                  
  │ trafo              │ Good                 │ No bugs found                                              │
  ├────────────────────┼──────────────────────┼────────────────────────────────────────────────────────────┤                                                  
  │ image_processing   │ Broken               │ Boundary handling and fast_box_blur algorithm wrong        │
  ├────────────────────┼──────────────────────┼────────────────────────────────────────────────────────────┤                                                  
  │ multimed           │ Broken               │ 3 critical bugs in core geometry                           │
  ├────────────────────┼──────────────────────┼────────────────────────────────────────────────────────────┤                                                  
  │ ray_tracing        │ OK                   │ Logic correct, just different variable naming              │
  ├────────────────────┼──────────────────────┼────────────────────────────────────────────────────────────┤                                                  
  │ imgcoord           │ Broken               │ Uses wrong camera center after transformation              │
  ├────────────────────┼──────────────────────┼────────────────────────────────────────────────────────────┤                                                  
  │ epi                │ Mostly OK            │ Error signaling differs (returns [] vs -1)                 │
  ├────────────────────┼──────────────────────┼────────────────────────────────────────────────────────────┤                                                  
  │ correspondences    │ Fundamentally broken │ Data structures incompatible, scoring formula wrong        │
  ├────────────────────┼──────────────────────┼────────────────────────────────────────────────────────────┤                                                  
  │ segmentation       │ Good                 │ Minor boundary differences                                 │
  ├────────────────────┼──────────────────────┼────────────────────────────────────────────────────────────┤                                                  
  │ sortgrid           │ Broken               │ Returns None objects, signature mismatches                 │
  ├────────────────────┼──────────────────────┼────────────────────────────────────────────────────────────┤                                                  
  │ orientation        │ Skeleton             │ ~90% unimplemented                                         │
  ├────────────────────┼──────────────────────┼────────────────────────────────────────────────────────────┤                                                  
  │ tracking_frame_buf │ Broken               │ Field names, array sizes, missing methods                  │
  ├────────────────────┼──────────────────────┼────────────────────────────────────────────────────────────┤                                                  
  │ tracking_run       │ Broken               │ Missing volumedimension, ymin/ymax                         │
  ├────────────────────┼──────────────────────┼────────────────────────────────────────────────────────────┤                                                  
  │ track              │ Mostly stubs         │ Core algorithms unimplemented                              │
  ├────────────────────┼──────────────────────┼────────────────────────────────────────────────────────────┤                                                  
  │ track3d            │ Broken               │ Field name crashes, missing fb methods                     │
  └────────────────────┴──────────────────────┴────────────────────────────────────────────────────────────┘                                                  
                  
  The most impactful fixes to prioritize: the multimed.py:296 typo (sr -= iz), the imgcoord.py wrong camera center, the correspondences.py scoring formula,   
  and the tracking_frame_buf.py field name mismatch. These are concrete bugs in otherwise-translated code. The stubs (track, orientation) need wholesale
  implementation.    
