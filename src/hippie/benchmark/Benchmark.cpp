/**
 * \file	Benchmark.cpp
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	Benchmark class source file.
 */

#include "Benchmark.h"
#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>

void Benchmark::testStatic() {

    QElapsedTimer timer;
    renderer.resetFrameIndex();

    float bvhKernelsTimeCur = MAX_FLOAT;
    float bvhAbsoluteTimeCur = MAX_FLOAT;
    float renderKernelsTimeCur = MAX_FLOAT;
    float renderAbsoluteTimeCur = MAX_FLOAT;
    float bvhKernelsTime = MAX_FLOAT;
    float bvhAbsoluteTime = MAX_FLOAT;
    float renderKernelsTime = MAX_FLOAT;
    float renderAbsoluteTime = MAX_FLOAT;
    float ttiKernels = MAX_FLOAT;
    float ttiAbsolute = MAX_FLOAT;
    float rtPerformance = -MAX_FLOAT;
    float rtPerformancePrimary = -MAX_FLOAT;
    float rtPerformanceShadow = -MAX_FLOAT;
    float rtPerformanceAo = -MAX_FLOAT;
    float rtPerformancePath = -MAX_FLOAT;
    float cost = MAX_FLOAT;
    unsigned long long numberOfRays = 0;
    unsigned long long numberOfPrimaryRays = 0;
    unsigned long long numberOfShadowRays = 0;
    unsigned long long numberOfAORays = 0;
    unsigned long long numberOfPathRays = 0;

    // Compute results.
    for (int i = 0; i < BENCHMARK_CYCLES; ++i) {
        timer.restart();
        timer.start();
        bvhKernelsTimeCur = builder->rebuild(*bvh);
        bvhKernelsTime = qMin(bvhKernelsTimeCur, bvhKernelsTime);
        //HipModule::sync();
        bvhAbsoluteTimeCur = 1.0e-3f * timer.elapsed();
        bvhAbsoluteTime = qMin(bvhAbsoluteTimeCur, bvhAbsoluteTime);
        cost = qMin(bvh->getCost(), cost);
        timer.restart();
        timer.start();
        renderer.resetFrameIndex();
        renderKernelsTimeCur = renderer.render(*scene, *bvh, camera, pixels, framePixels);
        renderKernelsTime = qMin(renderKernelsTimeCur, renderKernelsTime);
        renderAbsoluteTimeCur = 1.0e-3f * timer.elapsed();
        renderAbsoluteTime = qMin(renderAbsoluteTimeCur, renderAbsoluteTime);
        ttiKernels = qMin(bvhKernelsTimeCur + renderKernelsTimeCur, ttiKernels);
        ttiAbsolute = qMin(bvhAbsoluteTimeCur + renderAbsoluteTimeCur, ttiAbsolute);
        rtPerformance = qMax(rtPerformance, renderer.getTracePerformance());
        rtPerformancePrimary = qMax(rtPerformancePrimary, renderer.getPrimaryTracePerformance());
        rtPerformanceShadow = qMax(rtPerformanceShadow, renderer.getShadowTracePerformance());
        rtPerformanceAo = qMax(rtPerformanceAo, renderer.getAOTracePerformance());
        rtPerformancePath = qMax(rtPerformancePath, renderer.getPathTracePerformance());
        numberOfRays = qMax(numberOfRays, renderer.getNumberOfRays());
        numberOfPrimaryRays = qMax(numberOfPrimaryRays, renderer.getNumberOfPrimaryRays());
        numberOfShadowRays = qMax(numberOfShadowRays, renderer.getNumberOfShadowRays());
        numberOfAORays = qMax(numberOfAORays, renderer.getNumberOfAORays());
        numberOfPathRays = qMax(numberOfPathRays, renderer.getNumberOfPathRays());
        qInfo() << "INFO <Benchmark> Benchmark cycle " << i << " has been finished.";
    }

    // Write results.
    out << "RAY TYPE\n";
    out << rayTypeToString(renderer.getRayType()) << "\n\n";
    out << "BVH CONSTRUCTION TIME KERNELS\n";
    out << bvhKernelsTime << "\n\n";
    out << "BVH CONSTRUCTION TIME ABSOLUTE\n";
    out << bvhAbsoluteTime << "\n\n";
    out << "CT\n";
    out << bvh->getCt() << "\n\n";
    out << "CI\n";
    out << bvh->getCi() << "\n\n";
    out << "BVH COST\n";
    out << cost << "\n\n";
    out << "AVG. LEAF SIZE\n";
    out << bvh->getAvgLeafSize() << "\n\n";
    out << "LEAF SIZE HISTOGRAM\n";
    for (int h = 0; h < bvh->getLeafSizeHistogram().size(); ++h)
        out << bvh->getLeafSizeHistogram()[h] << "\n";
    out << "\n";

    out << "NUMBER OF RAYS\n";
    out << numberOfRays << "\n\n";
    out << "NUMBER OF PRIMARY RAYS\n";
    out << numberOfPrimaryRays << "\n\n";
    out << "NUMBER OF SHADOW RAYS\n";
    out << numberOfShadowRays << "\n\n";
    out << "NUMBER OF AO RAYS\n";
    out << numberOfAORays << "\n\n";
    out << "NUMBER OF PATH RAYS\n";
    out << numberOfPathRays << "\n\n";
    out << "RT PERFORMANCE\n";
    out << rtPerformance << "\n\n";
    out << "RT PERFORMANCE PRIMARY\n";
    out << rtPerformancePrimary << "\n\n";
    out << "RT PERFORMANCE SHADOW\n";
    out << rtPerformanceShadow << "\n\n";
    out << "RT PERFORMANCE AO\n";
    out << rtPerformanceAo << "\n\n";
    out << "RT PERFORMANCE PATH\n";
    out << rtPerformancePath << "\n\n";
    out << "RENDER TIME KERNELS\n";
    out << renderKernelsTime << "\n\n";
    out << "RENDER TIME ABSOLUTE\n";
    out << renderAbsoluteTime << "\n\n";
    out << "TIME TO IMAGE KERNELS\n";
    out << ttiKernels << "\n\n";
    out << "TIME TO IMAGE ABSOLUTE\n";
    out << ttiAbsolute << "\n";

    // Export image.
    if (images) {
        QString img = BENCHMARK_PICTURE_PREFIX + QString(".png");
        exporter.exportImage(camera.getSize().x, camera.getSize().y, framePixels, root + "/" + img, true);
        qInfo() << "INFO <Benchmark> Image " << img << " has been exported.";
    }

}

void Benchmark::testDynamic() {

    DynamicScene * dynamicScene = dynamic_cast<DynamicScene*>(scene);
    Q_ASSERT(dynamicScene != nullptr);

    QElapsedTimer timer;

    int numberOfFrames = ceil(dynamicScene->getFrameRate());

    float * cost = new float[numberOfFrames];
    float * sceneUpdateKernelsTime = new float[numberOfFrames];
    float * sceneUpdateAbsoluteTime = new float[numberOfFrames];
    float * bvhKernelsTime = new float[numberOfFrames];
    float * bvhAbsoluteTime = new float[numberOfFrames];
    float * renderKernelsTime = new float[numberOfFrames];
    float * renderAbsoluteTime = new float[numberOfFrames];
    float * ttiKernels = new float[numberOfFrames];
    float * ttiAbsolute = new float[numberOfFrames];
    float * rtPerformance = new float[numberOfFrames];
    float * rtPerformancePrimary = new float[numberOfFrames];
    float * rtPerformanceShadow = new float[numberOfFrames];
    float * rtPerformanceAO = new float[numberOfFrames];
    float * rtPerformancePath = new float[numberOfFrames];
    unsigned long long * numberOfRays = new unsigned long long[numberOfFrames];
    unsigned long long * numberOfPrimaryRays = new unsigned long long[numberOfFrames];
    unsigned long long * numberOfShadowRays = new unsigned long long[numberOfFrames];
    unsigned long long * numberOfAORays = new unsigned long long[numberOfFrames];
    unsigned long long * numberOfPathRays = new unsigned long long[numberOfFrames];

    float sceneUpdateKernelsTimeCur = 0.0f;
    float sceneUpdateAbsoluteTimeCur = 0.0f;
    float bvhKernelsTimeCur = MAX_FLOAT;
    float bvhAbsoluteTimeCur = MAX_FLOAT;
    float renderKernelsTimeCur = MAX_FLOAT;
    float renderAbsoluteTimeCur = MAX_FLOAT;

    for (int i = 0; i < numberOfFrames; ++i) {
        cost[i] = MAX_FLOAT;
        bvhKernelsTime[i] = MAX_FLOAT;
        bvhAbsoluteTime[i] = MAX_FLOAT;
        renderKernelsTime[i] = MAX_FLOAT;
        renderAbsoluteTime[i] = MAX_FLOAT;
        ttiKernels[i] = MAX_FLOAT;
        ttiAbsolute[i] = MAX_FLOAT;
        rtPerformance[i] = 0.0f;
        rtPerformanceShadow[i] = 0.0f;
        rtPerformanceAO[i] = 0.0f;
        rtPerformancePath[i] = 0.0f;
        numberOfRays[i] = 0;
        numberOfShadowRays[i] = 0;
        numberOfAORays[i] = 0;
        numberOfPathRays[i] = 0;
        if (i > 0) {
            sceneUpdateKernelsTime[i] = MAX_FLOAT;
            sceneUpdateAbsoluteTime[i] = MAX_FLOAT;
        }
        else {
            sceneUpdateKernelsTime[i] = 0.0f;
            sceneUpdateAbsoluteTime[i] = 0.0f;
        }
    }

    // Compute results.
    for (int i = 0; i < BENCHMARK_CYCLES; ++i) {

        sceneUpdateKernelsTimeCur = 0.0f;
        sceneUpdateAbsoluteTimeCur = 0.0f;

        dynamicScene->resetTime();
        dynamicScene->setFrameIndex(0);
        renderer.resetFrameIndex();
        bvh = builder->build(scene);

        for (int j = 0; j < numberOfFrames; ++j) {

            // Reset renderer.
            renderer.resetFrameIndex();

            // Update geometry (Skip for the first frame).
            if (j > 0) {
                timer.restart();
                timer.start();
                sceneUpdateKernelsTimeCur = interpolator.update(*dynamicScene);
                sceneUpdateAbsoluteTimeCur = 1.0e-3f * timer.elapsed();
                sceneUpdateKernelsTime[j] = qMin(sceneUpdateKernelsTimeCur, sceneUpdateKernelsTime[j]);
                sceneUpdateAbsoluteTime[j] = qMin(sceneUpdateAbsoluteTimeCur, sceneUpdateAbsoluteTime[j]);
            }

            // Update BVH.
            timer.restart();
            timer.start();
            if (updateMethod == REFIT) bvhKernelsTimeCur = bvh->update();
            else bvhKernelsTimeCur = builder->rebuild(*bvh);
            bvhKernelsTime[j] = qMin(bvhKernelsTimeCur, bvhKernelsTime[j]);
            bvhAbsoluteTimeCur = 1.0e-3f * timer.elapsed();
            bvhAbsoluteTime[j] = qMin(bvhAbsoluteTimeCur, bvhAbsoluteTime[j]);
            cost[j] = qMin(bvh->getCost(), cost[j]);

            // Render frame.
            timer.restart();
            timer.start();
            renderKernelsTimeCur = renderer.render(*scene, *bvh, camera, pixels, framePixels);
            renderAbsoluteTimeCur = 1.0e-3f * timer.elapsed();
            renderKernelsTime[j] = qMin(renderKernelsTimeCur, renderKernelsTime[j]);
            renderAbsoluteTime[j] = qMin(renderAbsoluteTimeCur, renderAbsoluteTime[j]);
            ttiKernels[j] = qMin(sceneUpdateKernelsTimeCur + bvhKernelsTimeCur + renderKernelsTimeCur, ttiKernels[j]);
            ttiAbsolute[j] = qMin(sceneUpdateAbsoluteTimeCur + bvhAbsoluteTimeCur + renderAbsoluteTimeCur, ttiAbsolute[j]);
            rtPerformance[j] = qMax(rtPerformance[j], renderer.getTracePerformance());
            rtPerformancePrimary[j] = qMax(rtPerformancePrimary[j], renderer.getPrimaryTracePerformance());
            rtPerformanceShadow[j] = qMax(rtPerformanceShadow[j], renderer.getShadowTracePerformance());
            rtPerformanceAO[j] = qMax(rtPerformanceAO[j], renderer.getAOTracePerformance());
            rtPerformancePath[j] = qMax(rtPerformancePath[j], renderer.getPathTracePerformance());
            numberOfRays[j] = qMax(numberOfRays[j], renderer.getNumberOfRays());
            numberOfPrimaryRays[j] = qMax(numberOfPrimaryRays[j], renderer.getNumberOfPrimaryRays());
            numberOfShadowRays[j] = qMax(numberOfShadowRays[j], renderer.getNumberOfShadowRays());
            numberOfAORays[j] = qMax(numberOfAORays[j], renderer.getNumberOfAORays());
            numberOfPathRays[j] = qMax(numberOfPathRays[j], renderer.getNumberOfPathRays());

            if (images && i == BENCHMARK_CYCLES - 1) {
                QString img = BENCHMARK_PICTURE_PREFIX + QString::number(j) + ".png";
                exporter.exportImage(camera.getSize().x, camera.getSize().y, framePixels, root + "/" + img, true);
                qInfo() << "INFO <Benchmark> Image " << img << " has been exported.";
            }

        }

        qInfo() << "INFO <Benchmark> Benchmark cycle " << i << " has been finished.";

    }

    // Maximum and average costs.
    float sceneUpdateKernelsTimeMax = 0.0f;
    float sceneUpdateKernelsTimeAvg = 0.0f;
    float sceneUpdateAbsoluteTimeMax = 0.0f;
    float sceneUpdateAbsoluteTimeAvg = 0.0f;
    float bvhKernelsTimeMax = 0.0f;
    float bvhKernelsTimeAvg = 0.0f;
    float bvhAbsoluteTimeMax = 0.0f;
    float bvhAbsoluteTimeAvg = 0.0f;
    float costMax = 0.0f;
    float costAvg = 0.0f;
    for (int i = 1; i < numberOfFrames; ++i) {
        sceneUpdateKernelsTimeMax = qMax(sceneUpdateKernelsTimeMax, sceneUpdateKernelsTime[i]);
        sceneUpdateKernelsTimeAvg += sceneUpdateKernelsTime[i];
        sceneUpdateAbsoluteTimeMax = qMax(sceneUpdateAbsoluteTimeMax, sceneUpdateAbsoluteTime[i]);
        sceneUpdateAbsoluteTimeAvg += sceneUpdateAbsoluteTime[i];
        bvhKernelsTimeMax = qMax(bvhKernelsTimeMax, bvhKernelsTime[i]);
        bvhKernelsTimeAvg += bvhKernelsTime[i];
        bvhAbsoluteTimeMax = qMax(bvhAbsoluteTimeMax, bvhAbsoluteTime[i]);
        bvhAbsoluteTimeAvg += bvhAbsoluteTime[i];
        costMax = qMax(costMax, cost[i]);
        costAvg += cost[i];
    }
    sceneUpdateKernelsTimeAvg /= (numberOfFrames - 1);
    sceneUpdateAbsoluteTimeAvg /= (numberOfFrames - 1);
    bvhKernelsTimeAvg /= (numberOfFrames - 1);
    bvhAbsoluteTimeAvg /= (numberOfFrames - 1);
    costAvg /= (numberOfFrames - 1);

    // Write BVH results.
    out << "RAY TYPE\n";
    out << rayTypeToString(renderer.getRayType()) << "\n\n";
    out << "MAX. SCENE UPDATE KERNELS\n";
    out << sceneUpdateKernelsTimeMax << "\n\n";
    out << "AVG. SCENE UPDATE KERNELS\n";
    out << sceneUpdateKernelsTimeAvg << "\n\n";
    out << "SCENE UPDATE KERNELS\n";
    for (int i = 1; i < numberOfFrames; ++i)
        out << sceneUpdateKernelsTime[i] << "\n";
    out << "\n";

    out << "MAX. SCENE UPDATE ABSOLUTE\n";
    out << sceneUpdateAbsoluteTimeMax << "\n\n";
    out << "AVG. SCENE UPDATE ABSOLUTE\n";
    out << sceneUpdateAbsoluteTimeAvg << "\n\n";
    out << "SCENE UPDATE ABSOLUTE\n";
    for (int i = 1; i < numberOfFrames; ++i)
        out << sceneUpdateAbsoluteTime[i] << "\n";
    out << "\n";

    out << "BVH UPDATE METHOD\n";
    out << updateMethodToString(updateMethod) << "\n\n";
    out << "MAX. BVH UPDATE TIME KERNELS\n";
    out << bvhKernelsTimeMax << "\n\n";
    out << "AVG. BVH UPDATE TIME KERNELS\n";
    out << bvhKernelsTimeAvg << "\n\n";
    out << "BVH UPDATE TIME KERNELS\n";
    for (int i = 1; i < numberOfFrames; ++i)
        out << bvhKernelsTime[i] << "\n";
    out << "\n";

    out << "MAX. BVH UDPATE TIME ABSOLUTE\n";
    out << bvhAbsoluteTimeMax << "\n\n";
    out << "AVG. BVH UPDATE TIME ABSOLUTE\n";
    out << bvhAbsoluteTimeAvg << "\n\n";
    out << "BVH UPDATE TIME ABSOLUTE\n";
    for (int i = 1; i < numberOfFrames; ++i)
        out << bvhAbsoluteTime[i] << "\n";
    out << "\n";

    out << "CT\n";
    out << bvh->getCt() << "\n\n";
    out << "CI\n";
    out << bvh->getCi() << "\n\n";
    out << "MAX. COST\n";
    out << costMax << "\n\n";
    out << "AVG. COST\n";
    out << costAvg << "\n\n";
    out << "BVH COST\n";
    for (int i = 1; i < numberOfFrames; ++i)
        out << cost[i] << "\n";
    out << "\n";
    out << "AVG. LEAF SIZE\n";
    out << bvh->getAvgLeafSize() << "\n\n";
    out << "LEAF SIZE HISTOGRAM\n";
    for (int h = 0; h < bvh->getLeafSizeHistogram().size(); ++h)
        out << bvh->getLeafSizeHistogram()[h] << "\n";
    out << "\n";

    // Compute average and extreme results.
    float renderAbsoluteTimeMax = 0.0f;
    float renderKernelsTimeMax = 0.0f;
    float ttiAbsoluteMax = 0.0f;
    float ttiKernelsMax = 0.0f;
    float renderAbsoluteTimeAvg = 0.0f;
    float renderKernelsTimeAvg = 0.0f;
    float ttiAbsoluteAvg = 0.0f;
    float ttiKernelsAvg = 0.0f;
    float rtPerformanceMin = MAX_FLOAT;
    float rtPerformanceAvg = 0.0f;
    float rtPerformancePrimaryMin = MAX_FLOAT;
    float rtPerformancePrimaryAvg = 0.0f;
    float rtPerformanceShadowMin = MAX_FLOAT;
    float rtPerformanceShadowAvg = 0.0f;
    float rtPerformanceAOMin = MAX_FLOAT;
    float rtPerformanceAOAvg = 0.0f;
    float rtPerformancePathMin = MAX_FLOAT;
    float rtPerformancePathAvg = 0.0f;
    for (int i = 1; i < numberOfFrames; ++i) {
        renderKernelsTimeMax = qMax(renderKernelsTimeMax, renderKernelsTime[i]);
        renderAbsoluteTimeMax = qMax(renderAbsoluteTimeMax, renderAbsoluteTime[i]);
        ttiKernelsMax = qMax(ttiKernelsMax, ttiKernels[i]);
        ttiAbsoluteMax = qMax(ttiAbsoluteMax, ttiAbsolute[i]);
        renderKernelsTimeAvg += renderKernelsTime[i];
        renderAbsoluteTimeAvg += renderAbsoluteTime[i];
        ttiKernelsAvg += ttiKernels[i];
        ttiAbsoluteAvg += ttiAbsolute[i];
        rtPerformanceAvg += rtPerformance[i];
        rtPerformancePrimaryAvg += rtPerformancePrimary[i];
        rtPerformanceShadowAvg += rtPerformanceShadow[i];
        rtPerformanceAOAvg += rtPerformanceAO[i];
        rtPerformancePathAvg += rtPerformancePath[i];
        rtPerformanceMin = qMin(rtPerformanceMin, rtPerformance[i]);
        rtPerformancePrimaryMin = qMin(rtPerformancePrimaryMin, rtPerformancePrimary[i]);
        rtPerformanceShadowMin = qMin(rtPerformanceShadowMin, rtPerformanceShadow[i]);
        rtPerformanceAOMin = qMin(rtPerformanceAOMin, rtPerformanceAO[i]);
        rtPerformancePathMin = qMin(rtPerformancePathMin, rtPerformancePath[i]);
    }
    renderKernelsTimeAvg /= (numberOfFrames - 1);
    renderAbsoluteTimeAvg /= (numberOfFrames - 1);
    ttiKernelsAvg /= (numberOfFrames - 1);
    ttiAbsoluteAvg /= (numberOfFrames - 1);
    rtPerformanceAvg /= (numberOfFrames - 1);
    rtPerformancePrimaryAvg /= (numberOfFrames - 1);
    rtPerformanceShadowAvg /= (numberOfFrames - 1);
    rtPerformanceAOAvg /= (numberOfFrames - 1);
    rtPerformancePathAvg /= (numberOfFrames - 1);

    // Ray results.
    out << "NUMBER OF RAYS\n";
    for (int i = 1; i < numberOfFrames; ++i)
        out << numberOfRays[i] << "\n";
    out << "\n";
    out << "NUMBER OF PRIMARY RAYS\n";
    for (int i = 1; i < numberOfFrames; ++i)
        out << numberOfPrimaryRays[i] << "\n";
    out << "\n";
    out << "NUMBER OF SHADOW RAYS\n";
    for (int i = 1; i < numberOfFrames; ++i)
        out << numberOfShadowRays[i] << "\n";
    out << "\n";
    out << "NUMBER OF AO RAYS\n";
    for (int i = 1; i < numberOfFrames; ++i)
        out << numberOfAORays[i] << "\n";
    out << "\n";
    out << "NUMBER OF PATH RAYS\n";
    for (int i = 1; i < numberOfFrames; ++i)
        out << numberOfPathRays[i] << "\n";
    out << "\n";

    // Ray tracing performance.
    out << "MIN. RT PERFORMANCE\n";
    out << rtPerformanceMin << "\n\n";
    out << "AVG. RT PERFORMANCE\n";
    out << rtPerformanceAvg << "\n\n";
    out << "RT PERFORMANCE\n";
    for (int i = 1; i < numberOfFrames; ++i)
        out << rtPerformance[i] << "\n";
    out << "\n";
    out << "MIN. RT PERFORMANCE PRIMARY\n";
    out << rtPerformancePrimaryMin << "\n\n";
    out << "AVG. RT PERFORMANCE PRIMARY\n";
    out << rtPerformancePrimaryAvg << "\n\n";
    out << "RT PERFORMANCE PRIMARY\n";
    for (int i = 1; i < numberOfFrames; ++i)
        out << rtPerformancePrimary[i] << "\n";
    out << "\n";
    out << "MIN. RT PERFORMANCE SHADOW\n";
    out << rtPerformanceShadowMin << "\n\n";
    out << "AVG. RT PERFORMANCE SHADOW\n";
    out << rtPerformanceShadowAvg << "\n\n";
    out << "RT PERFORMANCE SHADOW\n";
    for (int i = 1; i < numberOfFrames; ++i)
        out << rtPerformanceShadow[i] << "\n";
    out << "\n";
    out << "MIN. RT PERFORMANCE AO\n";
    out << rtPerformanceAOMin << "\n\n";
    out << "AVG. RT PERFORMANCE AO\n";
    out << rtPerformanceAOAvg << "\n\n";
    out << "RT PERFORMANCE AO\n";
    for (int i = 1; i < numberOfFrames; ++i)
        out << rtPerformanceAO[i] << "\n";
    out << "\n";
    out << "MIN. RT PERFORMANCE PATH\n";
    out << rtPerformancePathMin << "\n\n";
    out << "AVG. RT PERFORMANCE PATH\n";
    out << rtPerformancePathAvg << "\n\n";
    out << "RT PERFORMANCE PATH\n";
    for (int i = 1; i < numberOfFrames; ++i)
        out << rtPerformancePath[i] << "\n";
    out << "\n";

    // Render times of kernels.
    out << "MAX. RENDER TIME KERNELS\n";
    out << renderKernelsTimeMax << "\n\n";
    out << "AVG. RENDER TIME KERNELS\n";
    out << renderKernelsTimeAvg << "\n\n";
    out << "RENDER TIME KERNELS\n";
    for (int i = 1; i < numberOfFrames; ++i)
        out << renderKernelsTime[i] << "\n";
    out << "\n";

    // Render times of whole process.
    out << "MAX. RENDER TIME ABSOLUTE\n";
    out << renderAbsoluteTimeMax << "\n\n";
    out << "AVG. RENDER TIME ABSOLUTE\n";
    out << renderAbsoluteTimeAvg << "\n\n";
    out << "RENDER TIME ABSOLUTE\n";
    for (int i = 1; i < numberOfFrames; ++i)
        out << renderAbsoluteTime[i] << "\n";
    out << "\n";

    // Time to image of kernels.
    out << "MAX. TIME TO IMAGE KERNELS\n";
    out << ttiKernelsMax << "\n\n";
    out << "AVG. TIME TO IMAGE KERNELS\n";
    out << ttiKernelsAvg << "\n\n";
    out << "TIME TO IMAGE KERNELS\n";
    for (int i = 1; i < numberOfFrames; ++i)
        out << ttiKernels[i] << "\n";
    out << "\n";

    // Time to image of whole process.
    out << "MAX. TIME TO IMAGE ABSOLUTE\n";
    out << ttiAbsoluteMax << "\n\n";
    out << "AVG. TIME TO IMAGE ABSOLUTE\n";
    out << ttiAbsoluteAvg << "\n\n";
    out << "TIME TO IMAGE ABSOLUTE\n";
    for (int i = 1; i < numberOfFrames; ++i)
        out << ttiAbsolute[i] << "\n";
    out << "\n";

    delete[] cost;
    delete[] sceneUpdateKernelsTime;
    delete[] sceneUpdateAbsoluteTime;
    delete[] bvhKernelsTime;
    delete[] bvhAbsoluteTime;
    delete[] renderKernelsTime;
    delete[] renderAbsoluteTime;
    delete[] ttiKernels;
    delete[] ttiAbsolute;
    delete[] rtPerformance;
    delete[] rtPerformancePrimary;
    delete[] rtPerformanceShadow;
    delete[] rtPerformanceAO;
    delete[] rtPerformancePath;
    delete[] numberOfRays;
    delete[] numberOfPrimaryRays;
    delete[] numberOfShadowRays;
    delete[] numberOfAORays;
    delete[] numberOfPathRays;

}

QString Benchmark::rayTypeToString(Renderer::RayType rayType) {
    if (rayType == Renderer::PRIMARY_RAYS)
        return "PRIMARY";
    else if (rayType == Renderer::AO_RAYS)
        return "AO";
    else if (rayType == Renderer::PATH_RAYS)
        return "PATH";
    else if (rayType == Renderer::SHADOW_RAYS)
        return "SHADOW";
    else if (rayType == Renderer::PSEUDOCOLOR_RAYS)
        return "PSEUDOCOLOR";
    else
        return "THERMAL";
}

QString Benchmark::updateMethodToString(BVHUpdateMethod updateMethod) {
    if (updateMethod == REFIT)
        return "REFIT";
    else
        return "REBUILD";
}

void Benchmark::init() {

    Vec2i size;
    QString staticSceneFilename, sceneFilefilter, bvhMethod, bvhUpdateMethod;
    QStringList dynamicSceneFilenames;

    // Export images?
    Environment::getInstance()->getBoolValue("Benchmark.images", images);

    // Create output directory.
    Environment::getInstance()->getStringValue("Benchmark.output", output);
    root = "benchmark/" + output;
    if (!QDir(root).exists())
        QDir().mkpath(root);

    // Resolution.
    Environment::getInstance()->getIntValue("Resolution.width", size.x);
    Environment::getInstance()->getIntValue("Resolution.height", size.y);
    camera.setSize(size);

    // BVH builder
    Environment::getInstance()->getStringValue("Bvh.method", bvhMethod);
    if (bvhMethod == "lbvh") {
        builder = &lbvhBuilder;
    }
    else if (bvhMethod == "hlbvh") {
        builder = &hlbvhBuilder;
    }
    else if (bvhMethod == "sbvh") {
        builder = &sbvhBuilder;
    }
    else if (bvhMethod == "ploc") {
        builder = &plocBuilder;
    }
    else if (bvhMethod == "atr") {
        builder = &atrBuilder;
    }
    else if (bvhMethod == "tr") {
        builder = &trBuilder;
    }
    else if (bvhMethod == "insertion") {
        builder = &insertionBuilder;
    }

    // Update method.
    Environment::getInstance()->getStringValue("Bvh.update", bvhUpdateMethod);
    if (bvhUpdateMethod == "refit")
        updateMethod = REFIT;
    else
        updateMethod = REBUILD;

    // Sceme files.
    Environment::getInstance()->getStringValue("Scene.filename", staticSceneFilename);
    Environment::getInstance()->getStringValue("Scene.filefilter", sceneFilefilter);

    // Scene filenames.
    QFileInfo sceneInfo = QFileInfo(QFile(sceneFilefilter));
    QDir sceneDir = sceneInfo.dir();
    QStringList sceneFilter(sceneInfo.fileName());
    dynamicSceneFilenames = sceneDir.entryList(sceneFilter);
    for (QStringList::iterator i = dynamicSceneFilenames.begin(); i != dynamicSceneFilenames.end(); ++i)
        *i = sceneInfo.absolutePath() + "/" + *i;

    // Open scene.
    if (!staticSceneFilename.isEmpty() || dynamicSceneFilenames.size() > 1) {

        // Static scene.
        if (!staticSceneFilename.isEmpty() && dynamicSceneFilenames.size() <= 1) {
            scene = sceneLoader.loadStaticScene(staticSceneFilename);
        }

        // Dynamic scnee.
        else if (dynamicSceneFilenames.size() > 1) {
            DynamicScene * dynamicScene = nullptr;
            if (staticSceneFilename.isEmpty()) dynamicScene = sceneLoader.loadDynamicScene(dynamicSceneFilenames);
            else dynamicScene = sceneLoader.loadDynamicScene(staticSceneFilename, dynamicSceneFilenames);
            scene = dynamicScene;
        }

        // BVH.
        bvh = builder->build(scene);

    }

    // No scene => Exit.
    else {
        logger(LOG_ERROR) << "ERROR <Benchmark> No scene specified!\n";
        exit(EXIT_FAILURE);
    }

}

Benchmark::Benchmark() : scene(nullptr), bvh(nullptr) {
}

Benchmark::~Benchmark() {
    if (bvh) delete bvh;
    if (scene) delete scene;
}

void Benchmark::run() {
    init();
    qInfo() << "INFO <Benchmark> Output: " << output;
    qInfo() << "INFO <Benchmark> Ray type: " << rayTypeToString(renderer.getRayType());
    out.setOut(root + "/test_" + output + ".log");
    if (!scene->isDynamic()) testStatic();
    else testDynamic();
}
