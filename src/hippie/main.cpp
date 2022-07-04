/**
 * \file	main.cpp
 * \author	Daniel Meister
 * \date	2014/05/10
 * \brief	Main function source file.
 */

#include <QtWidgets/QApplication>
#include <QFile>
#include <QTextStream>
#include "benchmark/Benchmark.h"
#include "gpu/HipModule.h"
#include "gpu/HipCompiler.h"
#include "gui/MainWindow.h"
#include "util/Logger.h"

 /// A main entry point of the application.
 /**
  * A main entry point of the application.
  * \param[in]	argc		The number of command line arguments.
  * \param[in]	argv		The command line arguments.
  * \return					The value indicating success of the program.
  */
int main(int argc, char * argv[]) {

    // Initilize a logger.
    logger.setOut("out.log");
    logger.setErr("err.log");

    // Application return value.
    int retval = 0;

    // Initialize HIP framework.
    HipModule::staticInit();

    // Initialize environment.
    Environment * env = new AppEnvironment();
    Environment::setInstance(env);

    // Parse environment file.
    if (!env->parse(argc, argv, false))
        logger(LOG_WARN) << "WARN <main> Parsing failed environment file!\n";

    // Check mode.
    QString mode;
    Environment::getInstance()->getStringValue("Application.mode", mode);

    if (mode == "benchmark") {

        // Benchmark mode.
        logger(LOG_INFO) << "INFO <main> Benchmark mode.\n";
        Benchmark benchmark;
        benchmark.run();

    }

    else {

        // Interactive mode.
        logger(LOG_INFO) << "INFO <main> Interactive mode.\n";

        // Suppose interactive mode.
        if (mode != "interactive")
            logger(LOG_WARN) << "WARN <main> Unknown appplication mode. Interactive mode selected.\n";

        // Qt application.
        logger(LOG_INFO) << "INFO <main> Application started.\n";
        QApplication a(argc, argv);

        // Main window.
        MainWindow *  window = new MainWindow();
        window->init();

        // Show main window.
        window->show();

        // Return value of Qt application.
        retval = a.exec();

        // Finalize main window.
        delete window;

    }

    // Finalize HIP framework.
    HipModule::staticDeinit();

    // Return the value.
    return retval;

}
