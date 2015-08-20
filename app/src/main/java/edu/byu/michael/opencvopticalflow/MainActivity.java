package edu.byu.michael.opencvopticalflow;

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.OpenCVLoader;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Point;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Scalar;
import org.opencv.video.Video;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "com.packtpub.masteringopencvandroid.chapter5.MainActivity";

    private static final int VIEW_MODE_KLT_TRACKER = 0;
    private static final int VIEW_MODE_OPTICAL_FLOW = 1;

    private int mViewMode;
    private Mat mRgba;
    private Mat mIntermediateMat;
    private Mat mGray;
    private Mat mPrevGray;

    MatOfPoint2f prevFeatures, nextFeatures;
    MatOfPoint features;

    MatOfByte status;
    MatOfFloat err;


    private CameraBridgeViewBase mOpenCvCameraView;


    static {
        if (!OpenCVLoader.initDebug()) {
            // Handle initialization error
        }
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.main_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);

    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    protected void onResume() {
        super.onResume();
        Log.i(TAG, "OpenCV loaded successfully");
        mOpenCvCameraView.enableView();
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();
        //noinspection SimplifiableIfStatement
        if (id == R.id.mItemPreviewOpticalFlow) {
            mViewMode = VIEW_MODE_OPTICAL_FLOW;
            resetVars();
        }
        else if (id == R.id.mItemPreviewKLT) {
            mViewMode = VIEW_MODE_KLT_TRACKER;
           resetVars();
        }
        return super.onOptionsItemSelected(item);
    }

    private void resetVars() {
        mPrevGray = new Mat(mGray.rows(), mGray.cols(), CvType.CV_8UC1);
        features = new MatOfPoint();
        prevFeatures = new MatOfPoint2f();
        nextFeatures = new MatOfPoint2f();
        status = new MatOfByte();
        err = new MatOfFloat();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mIntermediateMat = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
        resetVars();
    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
        mIntermediateMat.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        final int viewMode = mViewMode;
        switch (viewMode) {
            case VIEW_MODE_OPTICAL_FLOW:
                mGray = inputFrame.gray();
                if (features.toArray().length == 0) {
                    int rowStep = 50, colStep = 100;
                    int nRows = mGray.rows() / rowStep, nCols = mGray.cols() / colStep;

                    Point points[] = new Point[nRows * nCols];
                    for (int i = 0; i < nRows; i++) {
                        for (int j = 0; j < nCols; j++) {
                            points[i * nCols + j] = new Point(j * colStep, i * rowStep);
                        }
                    }

                    features.fromArray(points);

                    prevFeatures.fromList(features.toList());
                    mPrevGray = mGray.clone();
                    break;
                }
                nextFeatures.fromArray(prevFeatures.toArray());
                Video.calcOpticalFlowPyrLK(mPrevGray, mGray, prevFeatures, nextFeatures, status, err);

                List<Point> prevList = features.toList(), nextList = nextFeatures.toList();
                Scalar color = new Scalar(255);

                for (int i = 0; i < prevList.size(); i++) {
                    Imgproc.line(mGray, prevList.get(i), nextList.get(i), color);

                }
                mPrevGray = mGray.clone();
                break;

            case VIEW_MODE_KLT_TRACKER:
                mGray = inputFrame.gray();

                if(features.toArray().length==0){
                    Imgproc.goodFeaturesToTrack(mGray, features, 10, 0.01, 10);
                    prevFeatures.fromList(features.toList());
                    mPrevGray = mGray.clone();
                    break;
                }

                Video.calcOpticalFlowPyrLK(mPrevGray, mGray, prevFeatures, nextFeatures, status, err);
                List<Point> drawFeature = nextFeatures.toList();
                for(int i = 0; i<drawFeature.size(); i++){
                    Point p = drawFeature.get(i);
                    Imgproc.circle(mGray, p, 5, new Scalar(255));
                }

                mPrevGray = mGray.clone();
                prevFeatures.fromList(nextFeatures.toList());
                break;
            default:
                mViewMode = VIEW_MODE_OPTICAL_FLOW;
        }
                return mGray;
        }
}

//More comments