import { App } from '@/components/app/app';

export default function Page() {
  const tokenServerId = process.env.LIVEKIT_TOKEN_SERVER_ID;
  const tokenEndpoint = '/api/token';
  const agentName = process.env.AGENT_NAME || 'my-agent';

  return (
    <App
      tokenServerId={tokenServerId}
      tokenEndpoint={tokenEndpoint}
      agentName={agentName}
      isVideoInputSupported={true}
    />
  );
}
