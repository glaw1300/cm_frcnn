import sys
sys.path.append("scripts")
from compare_base import *
from faster_rcnn import CNN
import utils
from datetime import datetime

# name of column with source ID's
SRC_ID_COL = "source_id"

"""
LOADING IN PLUME LIST
"""
# load in plume list
    # 2019
PLUME_LIST = "../permian_plume_list_EST_03252021.xlsx"
    # 2020
#PLUME_LIST = "../GAO_Summer2020_plume_source_list_multimodal.csv"
if PLUME_LIST.endswith(".csv"):
    df = pd.read_csv(PLUME_LIST)
else:
    df = pd.read_excel(PLUME_LIST)
# fill in missing sources
    # 2019
df["source_type"] = df["source_type"].fillna("NA")
    # 2020
#df = df[(df["lat"] != -9999) & (df["lon"] != -9999) & (~df[SRC_ID_COL].isna())]

"""
CNN config (all defaults by default)
"""
# set up CNN (with default parameters)
cnn = CNN()
cnn.setup_model()

"""
LOADING IN/MAKING PREDICTIONS
"""
# predictions from the CNN can be loaded in as dict from a pickle file
# {src_id (str) : ( tensor([Boxes[x1, y1, x2, y2]...]), tensor(scores), tensor(classes))}
PATH_TO_TIFS = "../tifs"
PICKLE_DIR = "data/pickles"
PREDS_PICKLE = "preds.pickle"
pred_path = os.path.join(PICKLE_DIR, PREDS_PICKLE)

# if preds saved, load them in
if os.path.exists(pred_path):
    print(f"Loading plumes from pickle: {PICKLE_DIR}/{PREDS_PICKLE}")
    with open(pred_path, "rb") as f:
        cnn_preds = pickle.load(f)
else:
    cnn_preds = {}
    # score threshold for prediction confidence
    SCORE_THRESH = .5
    # group images by source
    for name, group in df.groupby([SRC_ID_COL]):
        # get image info, if are none, exclude
        sid = int(name.lstrip("P"))

        tifs = glob.glob(os.path.join(PATH_TO_TIFS, str(sid)+"_*.tif"))
        if len(tifs) == 0:
            continue

        # if fast, only predict on the "least blurry" image (will select the all black imgs)
        if fast:
            tifs = [get_least_blurry_img(tifs)]

        # make predictions and store to dict
        cnn_preds[name] = get_prediction_outputs(tifs, cnn, SCORE_THRESH)

    # save predictions
    with open(pred_path, "wb") as f:
        pickle.dump(cnn_preds, f)

"""
LOADING IN / LOCATING PLUMES
"""
# load in plumes
PLUME_PICKLE = "plumes.pickle"
plume_path = os.path.join(PICKLE_DIR, PLUME_PICKLE)

# plumes need to be loaded in as a dictionary:
# {src_id (int) : ([(plume_x_pix, plume_y_pix), ...], [plume_id,...], [(q, qsigma),...])}
# if plumes saved in pickle, load them in
if os.path.exists(plume_path):
    print(f"Loading plumes from pickle: {PICKLE_DIR}/{PLUME_PICKLE}")
    with open(plume_path, "rb") as f:
        plumes = pickle.load(f)
else:
    plumes = {}
    print("Determining plume location in images")
    # group plumes by source id
    for name, group in tqdm.tqdm(df.groupby([SRC_ID_COL])):
        # get all images for given source
        sid = int(name.lstrip("P"))
        tifs = glob.glob(os.path.join(PATH_TO_TIFS, str(sid)+"_*.tif"))

        # if no images, continue
        if len(tifs) == 0:
            continue

        # save run of get_plume_info in plume dict
        plumes[sid] = get_plume_info(group, tifs[0])
    # save plume_dict for future use
    with open(plume_path, "wb") as f:
        pickle.dump(plumes, f)

"""
ITERATE OVER SOURCES, DETERMINE SOURCE FROM PREDS and PLUMES
"""
# TODO: to customize the source attribution algorithm, overwrite this method
#def determine_source(bboxes, scores, classes, loc, labels):
#    pass

ITERS = 100
# if you want to compare prediction to ground truth, give column name of source type
EXCLUDE = ["pipeline", "NA"]
    # 2019
SRC_TYPE_COL = "source_type"
    # 2020
#SRC_TYPE_COL = None
# directory to save results in
final_dir = datetime.now().strftime("%m%d%Y_%H:%m:%S")
os.makedirs(os.path.join(PICKLE_DIR, final_dir), exist_ok = True)
# do multiple iterations with
for n in range(ITERS):
    if SRC_ID_COL:
        success = 0
        fail = 0
    print(f"Beginning iteration {n} out of {ITERS}")
    # predictions stored in a dictionary:
    # {sourceid_plume# : (plumex, plumey, plumeid, tif, prediction, ground truth, distance, score)}
    preds = {}
    # iterate over each source
    for name, group in tqdm.tqdm(df.groupby([SRC_ID_COL])):
        # get image info, if there is none, continue
        sid = int(name.lstrip("P"))
        if sid in plumes:
            cents, pids, psizes = plumes[sid]
        else:
            continue

        # get all images, if no images, continue
        tifs = glob.glob(os.path.join(PATH_TO_TIFS, str(sid)+"_*.tif"))
        if len(tifs) == 0:
            continue

        # if no valid plumes, continue
        if len(cents) == 0:
            continue

        # if we are comparing to a ground truth, get the ground truth now
        if SRC_TYPE_COL:
            gt_src = group[SRC_TYPE_COL].unique()[0]
            # if a source to exclude, like a pipeline, ignore
            if gt_src in EXCLUDE:
                preds[f'{sid}'] = (None,)*5 + (gt_src,) + (None,)*2
                continue
        else:
            gt_src = "default"

        # if no predictions, continue
        bboxes, scores, classes = cnn_preds[name]
        if len(bboxes) == 0:
            preds[f'{sid}'] = (None,)*8

        # for each plume, get the mose likely prediction and save it to preds
        for i, ((x,y), pid, psize) in enumerate(zip(cents, pids, psizes)):
            # add noise to x and y b/c location of plume can vary
            x += np.random.normal(scale=30)
            y += np.random.normal(scale=30)

            # TODO: can change determine source method to customize source attribution
            # current method: min(distance/score)
            pred_src, d, s = determine_source(bboxes, scores, classes, (x, y), cnn.get_labels())

            # print current stats and store prediction
            store = (x, y, pid, tifs[0], pred_src, gt_src, d, s)
            preds[f"{sid}_{i}_{n}"] = store

            # if have a source type, do a comparison
            if SRC_TYPE_COL:
                # if match, mark as success
                if pred_src == gt_src:
                    success += 1
                else:
                    fail += 1

    # at the end of the iteration, publish accuracy
    if SRC_TYPE_COL:
        print(f"Accuracy: {success/(fail+success)}; {success} true, {fail} false")

    # write results to pickle
    with open(os.path.join(PICKLE_DIR, final_dir, f"source_attr_{n}.pickle"), "wb") as f:
        pickle.dump(preds, f)
