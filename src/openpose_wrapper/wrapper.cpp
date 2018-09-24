#include <utility>

#include "wrapper.h"

#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/headers.hpp>
#include <openpose/utilities/headers.hpp>
#include <openpose/core/array.hpp>
#include <openpose/core/point.hpp>
#include <openpose/utilities/keypoint.hpp>

using namespace ::std;

// Keep all loaded functions in here to prevent expensive construction/destruction calls.
struct State {
    State(op::ScaleAndSizeExtractor *scaleAndSizeExtractor,
            op::CvMatToOpInput *cvMatToOpInput,
            op::CvMatToOpOutput *cvMatToOpOutput,
            shared_ptr<op::PoseExtractorCaffe> poseExtractorCaffe,
            op::PoseRenderer *poseRenderer,
            op::OpOutputToCvMat *opOutputToCvMat) :
            scaleAndSizeExtractor(scaleAndSizeExtractor), cvMatToOpInput(cvMatToOpInput),
            cvMatToOpOutput(cvMatToOpOutput), poseExtractorCaffe(move(poseExtractorCaffe)),
            poseRenderer(poseRenderer), opOutputToCvMat(opOutputToCvMat) {}

    inline ~State() {
        delete scaleAndSizeExtractor;
        delete cvMatToOpInput;
        delete cvMatToOpOutput;
        delete poseRenderer;
        delete opOutputToCvMat;
    }

    op::ScaleAndSizeExtractor *scaleAndSizeExtractor;
    op::CvMatToOpInput *cvMatToOpInput;
    op::CvMatToOpOutput *cvMatToOpOutput;
    shared_ptr<op::PoseExtractorCaffe> poseExtractorCaffe;
    op::PoseRenderer *poseRenderer;
    op::OpOutputToCvMat *opOutputToCvMat;
};

static const int defaultInputWidth = 368;
static const int defaultGpuStartNumber = 0;
static const bool defaultUseGpus = true;

extern "C" {
    /**
     *
     * @param modelFolder Location containing the BODY_25 model for use by OpenPose
     * @param target_resolution_height Target resolution height must be of size %16==0 and greater than 0. Width will
     *      auto-adjust depending on height.
     * @param gpuStartNumber Which GPU to use in a multi-GPU system. Default to 0, meaning the first one.
     * @param useGpus If false, use CPU only.
     * @return
     */
    State* init(string* modelFolder, int target_resolution_height = defaultInputWidth,
                int gpuStartNumber = defaultGpuStartNumber, bool useGpus = defaultUseGpus);


    /**
     *
     * @param detector
     * @param inputImage
     * @return Matrix containing locations of all of your stuff. TODO Figure out exact format.
     */
    cv::Mat getPose(State* detector, cv::Mat inputImage);

    /**
     *
     * @param state
     * @param input
     * @return
     */
    cv::Mat getRenderedOutput(State *state, cv::Mat inputImage);

    /**
     * @param state Destroys the state and releases any resources being used.
     */
    int close(State* state);
}

static const int defaultInputHeight = -1;
static const op::Point<int> defaultOutput = op::Point<int>{-1, -1}; // Force output to be same size as input
static const op::PoseModel defaultPoseModel = op::PoseModel(0);  // BODY_25, the most accurate model
static const int defaultScaleNumber = 1;  // Number of scales to average. Can increase this for more passes across the data at the cost of speed.
static const double defaultScaleGap = 0.3;
static const double defaultRenderThreshold = 0.05;
static const double defaultAlphaPose = 0.6;
static const bool defaultEnableBlending = true;

// Initialize a new OpenPose detector.
State* init(const string &modelFolder, const int target_resolution_height, const int gpuStartNumber,
        const bool useGpus) {
    if (target_resolution_height % 16 != 0 || target_resolution_height < 1) {
        throw string("Target resolution height must be of size %16==0 and greater than 0.");
    }
    op::Point<int> netInputSize = op::Point<int>{defaultInputHeight, target_resolution_height};

    // Initialize all required classes.
    auto *scaleAndSizeExtractor = new op::ScaleAndSizeExtractor(netInputSize, defaultOutput, defaultScaleNumber,
                                                                defaultScaleGap);
    auto *cvMatToOpInput = new op::CvMatToOpInput(defaultPoseModel);
    auto *cvMatToOpOutput = new op::CvMatToOpOutput();
    shared_ptr<op::PoseExtractorCaffe> poseExtractorCaffe = make_shared<op::PoseExtractorCaffe>(
            op::PoseExtractorCaffe(defaultPoseModel, modelFolder, gpuStartNumber));
    op::PoseRenderer *poseRenderer;
    if (useGpus) {
        poseRenderer = new op::PoseGpuRenderer(defaultPoseModel, poseExtractorCaffe, (float) defaultRenderThreshold,
                                               defaultEnableBlending, (float) defaultAlphaPose);
    } else {
        poseRenderer = new op::PoseCpuRenderer(defaultPoseModel, (float) defaultRenderThreshold, defaultEnableBlending,
                                               (float) defaultAlphaPose);
    }
    auto *opOutputToCvMat = new op::OpOutputToCvMat();

    // Initialize actual resources.
    poseExtractorCaffe->initializationOnThread();
    poseRenderer->initializationOnThread();

    // Return state
    State *newOpenPoseDetector = new State{
            scaleAndSizeExtractor,
            cvMatToOpInput,
            cvMatToOpOutput,
            poseExtractorCaffe,
            poseRenderer,
            opOutputToCvMat
    };
    return newOpenPoseDetector;
}

cv::Mat getPose(State* detector, cv::Mat inputImage) {
    // TODO remove duplicate code
    // Step 1 - Read and load image, error if empty (possibly wrong path)
    if (inputImage.empty())
        throw string("Could not open or find the image");

    const op::Point<int> imageSize{inputImage.cols, inputImage.rows};

    // Step 2 - Get desired scale sizes
    vector<double> scaleInputToNetInputs;
    vector<op::Point<int>> netInputSizes;
    double scaleInputToOutput;
    op::Point<int> outputResolution;
    tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution) = detector->scaleAndSizeExtractor->
            extract(imageSize);

    // Step 3 - Format input image to OpenPose input and output formats
    const vector<op::Array<float>> netInputArray = detector->cvMatToOpInput->
            createArray(inputImage, scaleInputToNetInputs, netInputSizes);
    op::Array<float> outputArray = detector->cvMatToOpOutput->
            createArray(inputImage, scaleInputToOutput, outputResolution);

    // Step 4 - Estimate poseKeypoints
    detector->poseExtractorCaffe->forwardPass(netInputArray, imageSize, scaleInputToNetInputs);
    const op::Array<float> poseKeypoints = detector->poseExtractorCaffe->getPoseKeypoints();
    op::Array<float> rescaledKeypoints = poseKeypoints.clone();
    op::scaleKeypoints(rescaledKeypoints, (float)scaleInputToOutput);

    return rescaledKeypoints.getCvMat();
}

cv::Mat getRenderedOutput(State *detector, cv::Mat inputImage) {
    // Step 1 - Read and load image, error if empty (possibly wrong path)
    if (inputImage.empty())
        throw string("Could not open or find the image");

    const op::Point<int> imageSize{inputImage.cols, inputImage.rows};

    // Step 2 - Get desired scale sizes
    vector<double> scaleInputToNetInputs;
    vector<op::Point<int>> netInputSizes;
    double scaleInputToOutput;
    op::Point<int> outputResolution;
    tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution) = detector->scaleAndSizeExtractor->
            extract(imageSize);

    // Step 3 - Format input image to OpenPose input and output formats
    const vector<op::Array<float>> netInputArray = detector->cvMatToOpInput->
            createArray(inputImage, scaleInputToNetInputs, netInputSizes);
    op::Array<float> outputArray = detector->cvMatToOpOutput->
            createArray(inputImage, scaleInputToOutput, outputResolution);

    // Step 4 - Estimate poseKeypoints
    detector->poseExtractorCaffe->forwardPass(netInputArray, imageSize, scaleInputToNetInputs);
    const op::Array<float> poseKeypoints = detector->poseExtractorCaffe->getPoseKeypoints();

    // Step 5 - Render poseKeypoints
    detector->poseRenderer->renderPose(outputArray, poseKeypoints, (float)scaleInputToOutput);

    // Step 6 - OpenPose output format to cv::Mat
    auto outputImage = detector->opOutputToCvMat->formatToCvMat(outputArray);

    return outputImage;
}

int close(State* state) {
    delete state;
    return 0;
}
