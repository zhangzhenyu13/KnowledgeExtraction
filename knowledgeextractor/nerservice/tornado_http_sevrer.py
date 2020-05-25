# this  module leverages the tornado to build a high efficiency http sever
import logging
logging.getLogger( __name__ ).setLevel(logging.INFO)

try:
    import tornado
except Exception as e:
    logging.error("tornado is uninstalled, please visit:\n https://github.com/tornadoweb/tornado")
    e.with_traceback()

from tornado.web import RequestHandler, Application
from tornado.httpserver import HTTPServer
import knowledgeextractor
import json

def create_tornado_app(processCore, debug):
    #processCore=self.processCore
    class KGEHandler(RequestHandler):
        
        async def post(self):
            query_argument = json.loads(self.request.body)
            results= await processCore(query_argument)

            results = json.dumps(results)
            self.set_header('Content-type', 'application/json')
            self.write(results)

        

    app = Application([
        (r"/methodCore", KGEHandler)
        ],
        debug=debug
    )
    return app

class KGEHTTPPServer(object):
    def __init__(self,config_file):
        self.config = knowledgeextractor.KGEConfig(config_file)
        
    async def processCore(self, data):
        raise NotImplementedError
    
    
    def start(self):
        app= create_tornado_app(self.processCore, self.config.debug)
        server=HTTPServer(app)
        server.listen(port=self.config.port, address=self.config.listen_ip)
        logging.info("\n*************{} service is running({}:{})*************\n"
            .format(self.config.ServiceName, self.config.listen_ip, self.config.port))
        tornado.ioloop.IOLoop.current().start()

    