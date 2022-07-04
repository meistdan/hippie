/**
 * \file	AppEnvironment.h
 * \author	Daniel Meister
 * \date	2014/08/19
 * \brief	AppEnvironment class header file.
 */

#ifndef _APP_ENVIRONMENT_H_
#define _APP_ENVIRONMENT_H_

#include "Environment.h"

class AppEnvironment : public Environment {

protected:

    void registerOptions(void);

public:

    AppEnvironment(void);

};


#endif /* _APP_ENVIRONMENT_H_ */
